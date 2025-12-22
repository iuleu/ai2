# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Dict, Optional

class VideoClassificationModel(nn.Module):
    # Модель для классификации видео, объединяющая визуальные, временные и, опционально, текстовые признаки.
    def __init__(self, 
                 visual_feature_dim: int = 2048,  # например, из ResNet
                 temporal_feature_dim: int = 256, # например, из LSTM
                 text_feature_dim: int = 768,     # например, из BERT
                 num_classes: int = 10,           # Количество категорий действий/событий
                 fusion_hidden_dim: int = 512,    # Скрытая размерность для слоев объединения
                 dropout: float = 0.5,
                 use_text_features: bool = False, # Флаг для включения/отключения текстовых признаков
                 device: str = "cpu"):
        #     visual_feature_dim: Размерность визуальных признаков на кадр.
        #     temporal_feature_dim: Размерность вектора временных признаков.
        #     text_feature_dim: Размерность вектора текстовых признаков.
        #     num_classes: Количество выходных классов.
        #     fusion_hidden_dim: Скрытая размерность для слоев объединения признаков.
        #     dropout: Сила дропаута для регуляризации.
        #     use_text_features: Включать ли текстовые признаки в модель.
        #     device: 'cpu' или 'cuda'.
        super(VideoClassificationModel, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_text_features = use_text_features
        self.num_classes = num_classes

        # Слои объединения для комбинации визуальных и временных признаков
        # Размер входа зависит от того, используются ли текстовые признаки
        fusion_input_dim = visual_feature_dim + temporal_feature_dim
        if self.use_text_features:
            fusion_input_dim += text_feature_dim

        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Финальный классификационный слой
        self.classifier = nn.Linear(fusion_hidden_dim // 2, num_classes)

    def forward(self, visual_features: torch.Tensor, 
                temporal_features: torch.Tensor, 
                text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Прямой проход модели.
        #     visual_features: Тензор формы (batch_size, visual_feature_dim).
        #     temporal_features: Тензор формы (batch_size, temporal_feature_dim).
        #     text_features: Опциональный тензор формы (batch_size, text_feature_dim).
        batch_size = visual_features.size(0)

        # Объединить признаки по размерности признаков
        if self.use_text_features:
            if text_features is None:
                raise ValueError("Text features were expected but not provided.")
            combined_features = torch.cat((visual_features, temporal_features, text_features), dim=1)
        else:
            combined_features = torch.cat((visual_features, temporal_features), dim=1)

        # Пропустить через слои объединения
        fused_features = self.fusion_layers(combined_features)

        # Получить финальные логиты
        logits = self.classifier(fused_features)

        return logits

# Пример использования
if __name__ == "__main__":
    # Параметры модели
    visual_dim = 2048  # ResNet
    temporal_dim = 256 # LSTM
    text_dim = 768     # BERT
    num_classes = 5    # Пример: 5 категорий действий
    batch_size = 2     # Пример размера батча

    # Без текстовых признаков
    print("--- Scenario 1: Model without Text Features ---")
    model_no_text = VideoClassificationModel(
        visual_feature_dim=visual_dim,
        temporal_feature_dim=temporal_dim,
        num_classes=num_classes,
        use_text_features=False
    )
    model_no_text.eval() # Установить в режим оценки для примера инференса

    # Создаем фиктивные батчи признаков
    fake_visual = torch.randn(batch_size, visual_dim)
    fake_temporal = torch.randn(batch_size, temporal_dim)

    with torch.no_grad(): # Отключаем вычисление градиентов для примера
        output_no_text = model_no_text(fake_visual, fake_temporal)
    
    print(f"Input: Visual ({fake_visual.shape}), Temporal ({fake_temporal.shape})")
    print(f"Output shape (no text): {output_no_text.shape}")
    print(f"Output (first sample): {output_no_text[0].tolist()}")


    # С текстовыми признаками
    print("\n--- Scenario 2: Model with Text Features ---")
    model_with_text = VideoClassificationModel(
        visual_feature_dim=visual_dim,
        temporal_feature_dim=temporal_dim,
        text_feature_dim=text_dim,
        num_classes=num_classes,
        use_text_features=True
    )
    model_with_text.eval()

    fake_text = torch.randn(batch_size, text_dim)

    with torch.no_grad():
        output_with_text = model_with_text(fake_visual, fake_temporal, fake_text)

    print(f"Input: Visual ({fake_visual.shape}), Temporal ({fake_temporal.shape}), Text ({fake_text.shape})")
    print(f"Output shape (with text): {output_with_text.shape}")
    print(f"Output (first sample): {output_with_text[0].tolist()}")

    # Сравнение количества параметров
    print(f"\nModel parameters (no text): {sum(p.numel() for p in model_no_text.parameters()):,}")
    print(f"Model parameters (with text): {sum(p.numel() for p in model_with_text.parameters()):,}")