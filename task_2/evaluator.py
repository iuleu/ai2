# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Optional
from model import VideoClassificationModel

def evaluate_model(model: VideoClassificationModel, 
                   test_loader: DataLoader, 
                   num_classes: int,
                   device: str = "cpu",
                   top_k: int = (1, 5)) -> Dict[str, float]:

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval() # Установить модель в режим оценки

    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad(): # Отключить вычисление градиента
        for batch in test_loader:
            if len(batch) == 3:
                visual_batch, temporal_batch, labels = batch
                text_batch = None
            elif len(batch) == 4:
                visual_batch, temporal_batch, text_batch, labels = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            # Перенос данных на устройство
            visual_batch = visual_batch.to(device)
            temporal_batch = temporal_batch.to(device)
            labels = labels.to(device)
            if text_batch is not None:
                text_batch = text_batch.to(device)

            if model.use_text_features:
                if text_batch is None:
                    raise ValueError("Model expects text features but batch does not contain them.")
                logits = model(visual_batch, temporal_batch, text_batch)
            else:
                logits = model(visual_batch, temporal_batch)

            # Сохранение для расчета метрик
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Получение предсказание
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)

    # Вычисление метрик
    metrics = {}

    accuracy = accuracy_score(all_labels, all_predictions)
    metrics["accuracy"] = float(accuracy)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(num_classes), zero_division=0
    )
    for i in range(num_classes):
        metrics[f"precision_class_{i}"] = float(precision[i])
        metrics[f"recall_class_{i}"] = float(recall[i])
        metrics[f"f1_class_{i}"] = float(f1[i])

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    metrics["precision_macro"] = float(macro_precision)
    metrics["recall_macro"] = float(macro_recall)
    metrics["f1_macro"] = float(macro_f1)

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='micro', zero_division=0
    )
    metrics["precision_micro"] = float(micro_precision)
    metrics["recall_micro"] = float(micro_recall)
    metrics["f1_micro"] = float(micro_f1)

    for k in top_k:
        if k <= num_classes:
            top_k_indices = np.argpartition(all_logits, -k, axis=1)[:, -k:]
            top_k_correct = np.any(top_k_indices == all_labels[:, None], axis=1)
            top_k_acc = np.mean(top_k_correct)
            metrics[f"top_{k}_accuracy"] = float(top_k_acc)
        else:
            print(f"Warning: k={k} is greater than num_classes={num_classes}. Skipping top-{k} accuracy.")

    return metrics

# Пример использования
if __name__ == "__main__":
    from trainer import SimpleVideoDataset # Импортируем датасет
    # Параметры модели и данных
    visual_dim = 2048
    temporal_dim = 256
    text_dim = 768
    num_classes = 5
    batch_size = 4
    num_samples_test = 20

    print("--- Evaluating Scenario 1: Model without Text Features ---")
    
    # Создаем фиктивные тестовые данные
    test_visual = [np.random.rand(visual_dim).astype(np.float32) for _ in range(num_samples_test)]
    test_temporal = [np.random.rand(temporal_dim).astype(np.float32) for _ in range(num_samples_test)]
    test_labels = np.random.randint(0, num_classes, size=num_samples_test).tolist()

    test_dataset = SimpleVideoDataset(test_visual, test_temporal, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Инициализируем и обучим
    model_no_text = VideoClassificationModel(
        visual_feature_dim=visual_dim,
        temporal_feature_dim=temporal_dim,
        num_classes=num_classes,
        use_text_features=False
    )

    # Оцениваем
    metrics_no_text = evaluate_model(
        model=model_no_text,
        test_loader=test_loader,
        num_classes=num_classes
    )

    print("Evaluation Metrics (No Text Features):")
    for key, value in metrics_no_text.items():
        print(f"  {key}: {value:.4f}")


    print("\n--- Evaluating Scenario 2: Model with Text Features ---")
    
    test_text = [np.random.rand(text_dim).astype(np.float32) for _ in range(num_samples_test)]
    test_dataset_txt = SimpleVideoDataset(test_visual, test_temporal, test_labels, test_text)
    test_loader_txt = DataLoader(test_dataset_txt, batch_size=batch_size, shuffle=False)

    # Инициализируем и обучим модель
    model_with_text = VideoClassificationModel(
        visual_feature_dim=visual_dim,
        temporal_feature_dim=temporal_dim,
        text_feature_dim=text_dim,
        num_classes=num_classes,
        use_text_features=True
    )

    # Оцениваем
    metrics_with_text = evaluate_model(
        model=model_with_text,
        test_loader=test_loader_txt,
        num_classes=num_classes,
        top_k=(1, 3)
    )

    print("Evaluation Metrics (With Text Features):")
    for key, value in metrics_with_text.items():
        print(f"  {key}: {value:.4f}")

    print("\nExample evaluation completed.")