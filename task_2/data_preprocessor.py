# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
import decord
from decord import VideoReader
from PIL import Image

class VideoDataPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224), fps: float = 1.0):
        self.target_size = target_size
        self.fps = fps
        # —тандартные значени€ ImageNet дл€ нормализации
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        try:
            vr = VideoReader(video_path, ctx=decord.cpu(0)) # »спользовать CPU 0
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            
            if video_fps == 0:
                print(f"Warning: Could not determine FPS for {video_path}. Skipping.")
                return []
            
            # –ассчитать интервал между кадрами дл€ достижени€ целевой частоты кадров
            frame_interval = max(1, int(video_fps / self.fps))
            
            frame_indices = list(range(0, total_frames, frame_interval))
            frames = vr.get_batch(frame_indices).asnumpy() # ѕолучить пакет кадров
            
            # Decord возвращает кадры в формате RGB по умолчанию
            # ѕреобразовать из (N, H, W, C) в список (H, W, C)
            frame_list = [frame for frame in frames]
            
            print(f"Extracted {len(frame_list)} frames from {video_path}")
            return frame_list
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return []

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        # ѕреобразовать массив numpy в изображение PIL дл€ более простой обработки
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        # »зменить размер
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # ѕреобразовать обратно в массив numpy
        frame_resized = np.array(img, dtype=np.float32) # ‘орма (H, W, C)
        
        # Ќормализовать: разделить на 255, чтобы получить [0, 1], затем применить стандартную нормализацию
        frame_normalized = frame_resized / 255.0
        frame_normalized = (frame_normalized - self.mean) / self.std
        
        # “ранспонировать из (H, W, C) в (C, H, W)
        frame_final = np.transpose(frame_normalized, (2, 0, 1))
        
        return frame_final.astype(np.float32)

    def preprocess_video(self, video_path: str) -> List[np.ndarray]:
        # «агружает видео, извлекает кадры и предварительно обрабатывает их.
        #     —писок предварительно обработанных кадров в виде массивов numpy (C, H, W).
        raw_frames = self.extract_frames(video_path)
        if not raw_frames:
            return []
        
        preprocessed_frames = []
        for frame in raw_frames:
            preprocessed_frame = self.preprocess_frame(frame)
            preprocessed_frames.append(preprocessed_frame)
        
        return preprocessed_frames

# ѕример использовани€
if __name__ == "__main__":
    # ѕуть к тестовому видео
    test_video_path = "dataset/test_video.mp4"
    
    if os.path.exists(test_video_path):
        preprocessor = VideoDataPreprocessor(target_size=(224, 224), fps=1.0)
        processed_frames = preprocessor.preprocess_video(test_video_path)
        
        if processed_frames:
            print(f"Successfully preprocessed {len(processed_frames)} frames.")
            print(f"Frame shape: {processed_frames[0].shape}")
        else:
            print("No frames were extracted.")
    else:
        print(f"Test video file not found: {test_video_path}")
        print("Please place a test video file at the specified path to run the example.")