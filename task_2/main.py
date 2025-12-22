# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

# Импорты из нашей структуры
from data_preprocessor import VideoDataPreprocessor
from feature_extractor import VisualFeatureExtractor, TemporalFeatureExtractor
from model import VideoClassificationModel
from trainer import SimpleVideoDataset, VideoTrainer
from evaluator import evaluate_model

def main():
    print("=== Starting Video Classification Pipeline ===")

    base_video_path = "task_2/test_video.mp4"
    num_samples_train = 10
    num_samples_val = 4
    num_samples_test = 6
    num_classes = 5 # Количество "псевдо-классов" для демонстрации

    target_size = (224, 224)
    fps = 0.5  # Извлекать 1 кадр на 2 секунды
    visual_feature_dim = 2048  # ResNet50
    temporal_feature_dim = 256 # LSTM
    text_feature_dim = 768     # BERT
    batch_size = 2
    num_epochs = 5
    learning_rate = 1e-4
    top_k = (1, 3)

    # Загрузка и предобработка видео
    print("\n--- Step 1: Loading and Preprocessing Videos ---")
    preprocessor = VideoDataPreprocessor(target_size=target_size, fps=fps)

    # Извлекаем кадры из базового видео
    if not os.path.exists(base_video_path):
        print(f"Base video file not found: {base_video_path}")
        print("Please place a test video file to run the example.")
        return

    print(f"Preprocessing base video: {base_video_path}")
    base_frames = preprocessor.preprocess_video(base_video_path)
    if not base_frames:
        print("Could not preprocess base video.")
        return
    print(f"Preprocessed {len(base_frames)} frames from base video.")

    # Извлечение признаков
    print("\n--- Step 2: Extracting Features ---")
    # Инициализируем извлекатели
    visual_extractor = VisualFeatureExtractor(model_name="resnet50")
    temporal_extractor = TemporalFeatureExtractor(input_dim=visual_feature_dim, hidden_dim=temporal_feature_dim)

    print("Extracting features for base video...")
    visual_features_base = visual_extractor.extract_features(base_frames)
    temporal_feature_base = temporal_extractor.extract_features(visual_features_base)

    # Проверим, что признаки получены
    if visual_features_base.size == 0 or temporal_feature_base.size == 0:
        print("Failed to extract visual or temporal features from base video.")
        return

    # Создание датасетов
    print("\n--- Step 3: Creating Datasets ---")
    all_visual_features = []
    all_temporal_features = []
    all_labels = []
    all_text_features = []

    for i in range(num_samples_train + num_samples_val + num_samples_test):
        all_visual_features.append(np.mean(visual_features_base, axis=0))
        all_temporal_features.append(temporal_feature_base)
        all_labels.append(i % num_classes) # Циклически назначаем метки
        all_text_features.append(np.random.rand(text_feature_dim).astype(np.float32))

    # Разделим на подмножества
    train_v = all_visual_features[:num_samples_train]
    val_v = all_visual_features[num_samples_train:num_samples_train+num_samples_val]
    test_v = all_visual_features[num_samples_train+num_samples_val:]

    train_t = all_temporal_features[:num_samples_train]
    val_t = all_temporal_features[num_samples_train:num_samples_train+num_samples_val]
    test_t = all_temporal_features[num_samples_train+num_samples_val:]

    train_l = all_labels[:num_samples_train]
    val_l = all_labels[num_samples_train:num_samples_train+num_samples_val]
    test_l = all_labels[num_samples_train+num_samples_val:]

    train_txt = all_text_features[:num_samples_train]
    val_txt = all_text_features[num_samples_train:num_samples_train+num_samples_val]
    test_txt = all_text_features[num_samples_train+num_samples_val:]

    # Создаем датасеты
    train_dataset = SimpleVideoDataset(train_v, train_t, train_l, train_txt)
    val_dataset = SimpleVideoDataset(val_v, val_t, val_l, val_txt)
    test_dataset = SimpleVideoDataset(test_v, test_t, test_l, test_txt)

    print(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Создание и обучение модели
    print("\n--- Step 4: Creating and Training Model ---")
    # Инициализируем модель
    model = VideoClassificationModel(
        visual_feature_dim=visual_feature_dim,
        temporal_feature_dim=temporal_feature_dim,
        text_feature_dim=text_feature_dim,
        num_classes=num_classes,
        use_text_features=True
    )

    trainer = VideoTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    # Обучаем
    trainer.train()

    # Оценка модели
    print("\n--- Step 5: Evaluating Model ---")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Оцениваем
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        num_classes=num_classes,
        top_k=top_k
    )

    print("Final Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Сохранение метрик
    metrics_output_path = "task_2/results/metrics.json"
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to {metrics_output_path}")

    # Анализ ошибок
    print("\n--- Step 6: Error Analysis (Example) ---")
    print("Per-Class Metrics (from evaluation):")
    for i in range(num_classes):
        prec = metrics.get(f"precision_class_{i}", 0)
        recall = metrics.get(f"recall_class_{i}", 0)
        f1 = metrics.get(f"f1_class_{i}", 0)
        print(f"  Class {i}: P={prec:.4f}, R={recall:.4f}, F1={f1:.4f}")

    print("\n=== Video Classification Pipeline Completed ===")


if __name__ == "__main__":
    main()