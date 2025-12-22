# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import os

class VisualFeatureExtractor:
    def __init__(self, model_name: str = "resnet50", device: str = "cpu"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.preprocessor = None

        if model_name == "resnet50":
            # Загрузить предобученную ResNet50
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Удалить последний классификационный слой, чтобы получить признаки
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval() # Установить в режим оценки
            self.model.to(self.device)

        elif model_name == "clip":
            # Загрузить предобученную модель CLIP и процессор
            clip_model_name = "openai/clip-vit-base-patch32" # Или другой вариант CLIP
            self.model = CLIPModel.from_pretrained(clip_model_name)
            self.preprocessor = CLIPProcessor.from_pretrained(clip_model_name)
            self.model.eval()
            self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        # Извлекает признаки из списка кадров.
        if not frames:
            return np.array([])

        # Подготовить батч
        if self.model_name == "resnet50":
            # Преобразовать список массивов numpy в один массив numpy, а затем в тензор
            batch_np = np.stack(frames) # Форма: (N, C, H, W)
            batch_tensor = torch.from_numpy(batch_np).to(self.device) # Отправить на устройство

            with torch.no_grad():
                features_tensor = self.model(batch_tensor)
                # Форма вывода ResNet перед последним слоем: (N, 2048, 1, 1)
                features_tensor = features_tensor.squeeze(-1).squeeze(-1) # (N, 2048)

        elif self.model_name == "clip":
            # Наши кадры (C, H, W) float32. CLIPProcessor ожидает (H, W, C) uint8 изображения PIL
            # Преобразовать наши нормализованные float кадры обратно в PIL для процессора CLIP
            pil_frames = []
            # Денормализовать, используя те же статистики, что и в VideoDataPreprocessor
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            for frame in frames:
                # форма кадра (C, H, W) -> (H, W, C)
                frame_hwc = frame.transpose(1, 2, 0)
                # Денормализовать
                frame_denorm = (frame_hwc * std) + mean
                # Ограничить значения [0, 1] и преобразовать в [0, 255] uint8
                frame_uint8 = np.clip(frame_denorm, 0, 1) * 255
                pil_frame = Image.fromarray(frame_uint8.astype(np.uint8))
                pil_frames.append(pil_frame)

            # Обработать изображения PIL с помощью процессора CLIP
            inputs = self.preprocessor(images=pil_frames, return_tensors="pt", padding=True)
            pixel_values = inputs.pixel_values.to(self.device)

            with torch.no_grad():
                features_tensor = self.model.get_image_features(pixel_values=pixel_values)
                # Форма признаков изображений CLIP: (N, feature_dim), например, (N, 768)

        features_np = features_tensor.cpu().numpy()
        return features_np

class TemporalFeatureExtractor:
    # Извлекает временные признаки с помощью моделей, таких как LSTM, применяемых к визуальным признакам.
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1, device: str = "cpu"):
        #     input_dim: Размерность входных признаков (например, 2048 из ResNet, 768 из CLIP).
        #     hidden_dim: Скрытая размерность LSTM.
        #     num_layers: Количество слоев LSTM.
        #     device: 'cpu' или 'cuda'.
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm.eval() # Установить в режим оценки
        self.lstm.to(self.device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def extract_features(self, visual_features: np.ndarray) -> np.ndarray:
        # Извлекает временные признаки из последовательности визуальных признаков.
        if len(visual_features) == 0:
            return np.array([])

        # Добавить размерность батча: (N, feature_dim) -> (1, N, feature_dim)
        batch_features = np.expand_dims(visual_features, axis=0)
        # Преобразовать в тензор и отправить на устройство
        features_tensor = torch.from_numpy(batch_features).float().to(self.device)

        with torch.no_grad():
            lstm_out, (h_n, c_n) = self.lstm(features_tensor)
            # Взять скрытое состояние последнего слоя
            temporal_feature = h_n[-1, 0, :].cpu().numpy()

        return temporal_feature

# Пример использования
if __name__ == "__main__":
    # Путь к тестовому видео
    test_video_path = "task_2/test_video.mp4"

    if os.path.exists(test_video_path):

        print("--- Testing Feature Extractor with Dummy Frames (simulating output from data_preprocessor) ---")
        # Имитируем 5 кадров, предобработанных VideoDataPreprocessor: (C=3, H=224, W=224), float32, normalized

        # ИМИТАЦИЯ ВЫЗОВА DATA_PREPROCESSOR
        import sys
        import os

        # Повторим логику VideoDataPreprocessor для теста
        import decord
        from decord import VideoReader
        from PIL import Image
        import numpy as np

        def dummy_preprocess_video(video_path, target_size=(224, 224), fps=1.0):
            """Imitates VideoDataPreprocessor logic."""
            try:
                vr = VideoReader(video_path, ctx=decord.cpu(0))
                total_frames = len(vr)
                video_fps = vr.get_avg_fps()
                if video_fps == 0: return []
                frame_interval = max(1, int(video_fps / fps))
                frame_indices = list(range(0, total_frames, frame_interval))
                frames = vr.get_batch(frame_indices).asnumpy()
                raw_frames = [frame for frame in frames]

                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])

                preprocessed_frames = []
                for frame in raw_frames:
                    img = Image.fromarray(frame.astype('uint8'), 'RGB')
                    img = img.resize(target_size, Image.Resampling.BILINEAR)
                    frame_resized = np.array(img, dtype=np.float32)
                    frame_normalized = frame_resized / 255.0
                    frame_normalized = (frame_normalized - mean) / std
                    frame_final = np.transpose(frame_normalized, (2, 0, 1))
                    preprocessed_frames.append(frame_final.astype(np.float32))
                return preprocessed_frames
            except Exception as e:
                print(f"Error in dummy preprocess: {e}")
                return []

        processed_frames = dummy_preprocess_video(test_video_path, fps=0.5)
        print(f"Dummy preprocessor loaded {len(processed_frames)} frames.")

        if processed_frames:
            print(f"Preprocessed {len(processed_frames)} frames.")

            # 2. Извлечение визуальных признаков (ResNet)
            print("\n--- Extracting Visual Features (ResNet50) ---")
            visual_extractor_resnet = VisualFeatureExtractor(model_name="resnet50")
            visual_features_resnet = visual_extractor_resnet.extract_features(processed_frames)
            print(f"Visual features shape (ResNet): {visual_features_resnet.shape}") # (N_frames, 2048)

            # 3. Извлечение визуальных признаков (CLIP) - может быть медленнее
            print("\n--- Extracting Visual Features (CLIP) ---")
            visual_extractor_clip = VisualFeatureExtractor(model_name="clip")
            visual_features_clip = visual_extractor_clip.extract_features(processed_frames)
            print(f"Visual features shape (CLIP): {visual_features_clip.shape}") # (N_frames, 768)

            # 4. Извлечение временных признаков (на примере ResNet features)
            print("\n--- Extracting Temporal Features (LSTM on ResNet feats) ---")
            temporal_extractor = TemporalFeatureExtractor(input_dim=visual_features_resnet.shape[1], hidden_dim=256) # 2048 for ResNet
            temporal_feature = temporal_extractor.extract_features(visual_features_resnet)
            print(f"Temporal feature shape: {temporal_feature.shape}") # (256,)

            # 5. Извлечение временных признаков (на примере CLIP features)
            print("\n--- Extracting Temporal Features (LSTM on CLIP feats) ---")
            temporal_extractor_clip = TemporalFeatureExtractor(input_dim=visual_features_clip.shape[1], hidden_dim=128) # 768 for CLIP
            temporal_feature_clip = temporal_extractor_clip.extract_features(visual_features_clip)
            print(f"Temporal feature shape (from CLIP): {temporal_feature_clip.shape}") # (128,)

        else:
            print("No frames were extracted for feature extraction.")
    else:
        print(f"Test video file not found: {test_video_path}")
        print("Please place a test video file at the specified path to run the example.")