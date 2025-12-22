# -*- coding: utf-8 -*-

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import numpy as np
from typing import List, Dict

def evaluate_model(y_true: List[int], y_pred: List[int], y_scores: List[float] = None) -> Dict:
    # Основные метрики
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    # ROC-AUC если вероятности предоставлены
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            metrics["roc_auc"] = float(auc)
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")
    
    return metrics

def analyze_features(importance_scores: np.ndarray, feature_names: List[str] = None, top_k: int = 10):
    # Анализ самых важных признаков (для интерпретации модели)
    indices = np.argsort(importance_scores)[::-1][:top_k]
    top_features = [(feature_names[i], importance_scores[i]) for i in indices] if feature_names else indices
    return top_features

# Тестирование
if __name__ == "__main__":
    # Пример оценки
    true_labels = [1, 0, 1, 1, 0]
    pred_labels = [1, 0, 0, 1, 1]
    scores = [0.9, 0.2, 0.4, 0.8, 0.6]
    
    metrics = evaluate_model(true_labels, pred_labels, scores)
    print("Metrics:", metrics)