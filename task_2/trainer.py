# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from model import VideoClassificationModel

class SimpleVideoDataset(Dataset):
    def __init__(self, visual_features: List[np.ndarray], 
                 temporal_features: List[np.ndarray],
                 labels: List[int],
                 text_features: Optional[List[np.ndarray]] = None):
        #     visual_features: Список векторов визуальных признаков (например, объединенные из кадров).
        #     temporal_features: Список векторов временных признаков.
        #     labels: Список целочисленных меток (индексы классов).
        #     text_features: Опциональный список векторов текстовых признаков.
        self.visual_features = visual_features
        self.temporal_features = temporal_features
        self.labels = labels
        self.text_features = text_features

        if text_features is not None:
            assert len(visual_features) == len(text_features), "Number of visual and text features must match."
        assert len(visual_features) == len(temporal_features) == len(labels), \
               "Number of visual, temporal features and labels must match."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        v_feat = torch.from_numpy(self.visual_features[idx]).float()
        t_feat = torch.from_numpy(self.temporal_features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.text_features is not None:
            txt_feat = torch.from_numpy(self.text_features[idx]).float()
            return v_feat, t_feat, txt_feat, label
        else:
            return v_feat, t_feat, label

class VideoTrainer:
    # Обрабатывает цикл обучения для VideoClassificationModel.
    def __init__(self, model: VideoClassificationModel, 
                 train_dataset: SimpleVideoDataset,
                 val_dataset: Optional[SimpleVideoDataset] = None,
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 10,
                 device: str = "cpu"):
        #     model: Модель для обучения.
        #     train_dataset: Обучающий датасет.
        #     val_dataset: Валидационный датасет (опционально).
        #     batch_size: Размер обучающих батчей.
        #     learning_rate: Скорость обучения для оптимизатора.
        #     num_epochs: Количество эпох для обучения.
        #     device: 'cpu' или 'cuda'.
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss() # Стандартная функция потерь для многоклассовой классификации

        # Использовать DataLoader для батчинга
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    def train_epoch(self) -> float:
        self.model.train() # Установить модель в режим обучения
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Распаковать батч
            if len(batch) == 3: # Без текстовых признаков
                visual_batch, temporal_batch, labels = batch
                text_batch = None
            elif len(batch) == 4: # С текстовыми признаками
                visual_batch, temporal_batch, text_batch, labels = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            # Переместить данные на устройство
            visual_batch = visual_batch.to(self.device)
            temporal_batch = temporal_batch.to(self.device)
            labels = labels.to(self.device)
            if text_batch is not None:
                text_batch = text_batch.to(self.device)

            # Обнулить градиенты
            self.optimizer.zero_grad()

            # Прямой проход
            if self.model.use_text_features:
                if text_batch is None:
                    raise ValueError("Model expects text features but batch does not contain them.")
                logits = self.model(visual_batch, temporal_batch, text_batch)
            else:
                logits = self.model(visual_batch, temporal_batch)

            # Рассчитать потерю
            loss = self.criterion(logits, labels)

            # Обратный проход
            loss.backward()

            # Обновить веса
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> Optional[Dict[str, float]]:
        # Выполняет одну валидационную эпоху.
        if self.val_loader is None:
            return None

        self.model.eval() # Установить модель в режим оценки
        total_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad(): # Отключить вычисление градиентов для валидации
            for batch in self.val_loader:
                if len(batch) == 3: # Без текстовых признаков
                    visual_batch, temporal_batch, labels = batch
                    text_batch = None
                elif len(batch) == 4: # С текстовыми признаками
                    visual_batch, temporal_batch, text_batch, labels = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                # Переместить данные на устройство
                visual_batch = visual_batch.to(self.device)
                temporal_batch = temporal_batch.to(self.device)
                labels = labels.to(self.device)
                if text_batch is not None:
                    text_batch = text_batch.to(self.device)

                # Прямой проход
                if self.model.use_text_features:
                    if text_batch is None:
                        raise ValueError("Model expects text features but batch does not contain them.")
                    logits = self.model(visual_batch, temporal_batch, text_batch)
                else:
                    logits = self.model(visual_batch, temporal_batch)

                # Рассчитать потерю
                loss = self.criterion(logits, labels)
                total_val_loss += loss.item()

                # Рассчитать точность
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                num_batches += 1

        avg_val_loss = total_val_loss / num_batches
        val_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return {
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        }

    def train(self):
        # Основной цикл обучения.
        print("Starting training...")
        for epoch in range(self.num_epochs):
            # Обучить одну эпоху
            train_loss = self.train_epoch()
            
            # Валидировать
            val_metrics = self.validate()

            # Вывести метрики
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}", end="")
            if val_metrics:
                print(f" - Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")
            else:
                print("") # Новая строка, если нет валидации

        print("Training finished.")

# Пример использования
if __name__ == "__main__":
    # Параметры модели
    visual_dim = 2048  # ResNet
    temporal_dim = 256 # LSTM
    text_dim = 768     # BERT
    num_classes = 5    # Пример: 5 категорий действий
    batch_size = 2     # Пример размера батча
    num_epochs = 3     # Мало эпох для демонстрации

    # Обучение без текстовых признаков
    print("--- Training Scenario 1: Model without Text Features ---")
    
    num_samples_train = 20
    num_samples_val = 10

    # Train data
    train_visual = [np.random.rand(visual_dim).astype(np.float32) for _ in range(num_samples_train)]
    train_temporal = [np.random.rand(temporal_dim).astype(np.float32) for _ in range(num_samples_train)]
    train_labels = np.random.randint(0, num_classes, size=num_samples_train).tolist()

    # Val data
    val_visual = [np.random.rand(visual_dim).astype(np.float32) for _ in range(num_samples_val)]
    val_temporal = [np.random.rand(temporal_dim).astype(np.float32) for _ in range(num_samples_val)]
    val_labels = np.random.randint(0, num_classes, size=num_samples_val).tolist()

    train_dataset = SimpleVideoDataset(train_visual, train_temporal, train_labels)
    val_dataset = SimpleVideoDataset(val_visual, val_temporal, val_labels)

    # Инициализируем модель
    model_no_text = VideoClassificationModel(
        visual_feature_dim=visual_dim,
        temporal_feature_dim=temporal_dim,
        num_classes=num_classes,
        use_text_features=False
    )

    # Инициализируем тренер
    trainer_no_text = VideoTrainer(
        model=model_no_text,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=1e-3
    )

    # Обучаем
    trainer_no_text.train()


    # Обучение с текстовыми признаками
    print("\n--- Training Scenario 2: Model with Text Features ---")
    
    # Train data (with text)
    train_text = [np.random.rand(text_dim).astype(np.float32) for _ in range(num_samples_train)]
    # Val data (with text)
    val_text = [np.random.rand(text_dim).astype(np.float32) for _ in range(num_samples_val)]

    train_dataset_txt = SimpleVideoDataset(train_visual, train_temporal, train_labels, train_text)
    val_dataset_txt = SimpleVideoDataset(val_visual, val_temporal, val_labels, val_text)

    # Инициализируем модель
    model_with_text = VideoClassificationModel(
        visual_feature_dim=visual_dim,
        temporal_feature_dim=temporal_dim,
        text_feature_dim=text_dim,
        num_classes=num_classes,
        use_text_features=True
    )

    # Инициализируем тренер
    trainer_with_text = VideoTrainer(
        model=model_with_text,
        train_dataset=train_dataset_txt,
        val_dataset=val_dataset_txt,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=1e-3
    )

    # Обучаем
    trainer_with_text.train()

    print("\nExample training completed.")