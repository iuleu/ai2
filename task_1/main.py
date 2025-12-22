# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_collector import create_dataset
from text_vectorizer import LLMVectorizer
from classifier import ZeroShotClassifier
from evaluator import evaluate_model
from trained_classifier import TrainedClassifier

def main():
    # 1. Создать набор данных
    print("=== Step 1: Creating dataset ===")
    dataset_path = "task_1/dataset/train_data.json"
    if not os.path.exists(dataset_path):
        create_dataset(dataset_path)
    
    # Загрузить данные
    import json
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    true_labels = [item['label'] for item in data]
    
    print(f"Loaded {len(texts)} texts")
    print(f"  - Real news: {sum(true_labels)}")
    print(f"  - Fake news: {len(true_labels) - sum(true_labels)}")
    
    # Классификация Zero-Shot
    print("\n=== Step 2: Zero-Shot Classification (full dataset) ===")
    classifier_zs = ZeroShotClassifier()
    predictions_zs = classifier_zs.predict(texts)
    
    # Преобразовать предсказания в двоичные метки для оценки
    pred_labels_zs = []
    pred_scores_zs = [] # Для ROC-AUC, оценка для положительного класса ('real'=1)
    label_map_zs = {"fake": 0, "real": 1}
    for pred in predictions_zs:
        pred_label = label_map_zs.get(pred["predicted_label"], 0)
        pred_labels_zs.append(pred_label)
        # Использовать оценку достоверности для метки 'real' для ROC-AUC
        real_confidence = pred["all_scores"].get("real", 0.5)
        pred_scores_zs.append(real_confidence)
    
    # Оценить классификатор Zero-Shot
    eval_metrics_zs = evaluate_model(true_labels, pred_labels_zs, pred_scores_zs)
    print("Zero-Shot Classifier Metrics:")
    for metric, value in eval_metrics_zs.items():
        print(f"  {metric}: {value:.4f}")

    # Обученный классификатор (на эмбеддингах LLM)
    print("\n=== Step 3: Training & Evaluating Trained Classifier (on LLM embeddings) ===")
    # Использовать LLMVectorizer для получения эмбеддингов для всего набора данных
    vectorizer = LLMVectorizer()
    embeddings_full = vectorizer.encode_texts(texts)
    print(f"Full dataset embedding shape: {embeddings_full.shape}")

    # Инициализировать и обучить классификатор с использованием эмбеддингов LLM
    classifier_trained = TrainedClassifier(vectorizer_type="llm") # Использовать эмбеддинги LLM
    classifier_trained.fit(texts, true_labels) # Обучить с использованием оригинальных текстов (метод fit обрабатывает эмбеддинги)

    # Предсказать с использованием обученного классификатора
    pred_labels_trained = classifier_trained.predict(texts)
    pred_probs_trained = classifier_trained.predict_proba(texts)
    pred_scores_trained = pred_probs_trained[:, 1]

    # Оценить обученный классификатор
    eval_metrics_trained = evaluate_model(true_labels, pred_labels_trained, pred_scores_trained)
    print("Trained Classifier Metrics (on LLM embeddings):")
    for metric, value in eval_metrics_trained.items():
        print(f"  {metric}: {value:.4f}")

    # Анализ признаков для обученного классификатора
    print("\n=== Step 4: Feature Analysis (Trained Classifier) ===")
    top_features = classifier_trained.analyze_top_features(top_k=10)
    print("Top 10 Most Important Embedding Dimensions (by coefficient magnitude):")
    for name, coef in top_features:
        print(f"  {name}: {coef:.4f}")

    # Сохранить обученную модель
    model_save_path = "task_1/models/trained_fake_news_model_llm.pkl"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    classifier_trained.save_model(model_save_path)


    # Вывести индивидуальные предсказания для просмотра (Zero-Shot)
    print("\n--- Sample Individual Zero-Shot Predictions ---")
    for i in range(min(3, len(texts))): # Вывести первые 3
        text, true_label, pred_item = texts[i], true_labels[i], predictions_zs[i]
        status = "CORRECT" if label_map_zs.get(pred_item["predicted_label"]) == true_label else "INCORRECT"
        print(f"[{i}] True: {'REAL' if true_label else 'FAKE'}, Pred: {pred_item['predicted_label']} ({pred_item['confidence']:.2f}), Status: {status}")
        print(f"     Text: {text[:100]}...") # Показать первые 100 символов текста


if __name__ == "__main__":
    main()