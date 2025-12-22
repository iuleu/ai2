# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Tuple, Optional
from text_vectorizer import LLMVectorizer
import joblib

class TrainedClassifier:
    def __init__(self, vectorizer_type: str = "llm"):
        self.vectorizer_type = vectorizer_type
        self.llm_vectorizer = None
        self.tfidf_vectorizer = None
        self.model = LogisticRegression(max_iter=1000) # Увеличить max_iter, чтобы избежать предупреждения о сходимости

        if vectorizer_type == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        elif vectorizer_type == "llm":
            self.llm_vectorizer = LLMVectorizer()
        else:
            raise ValueError("Unsupported vectorizer type. Use 'tfidf' or 'llm'.")

    def fit(self, texts: List[str], labels: List[int]):
        if self.vectorizer_type == "tfidf":
            X = self.tfidf_vectorizer.fit_transform(texts)
        elif self.vectorizer_type == "llm":
            # Векторизовать тексты с использованием LLM
            X_array = self.llm_vectorizer.encode_texts(texts)
            X = X_array

        self.model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        if self.vectorizer_type == "tfidf":
            X = self.tfidf_vectorizer.transform(texts)
        elif self.vectorizer_type == "llm":
            X_array = self.llm_vectorizer.encode_texts(texts)
            X = X_array

        return self.model.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer_type == "tfidf":
            X = self.tfidf_vectorizer.transform(texts)
        elif self.vectorizer_type == "llm":
            X_array = self.llm_vectorizer.encode_texts(texts)
            X = X_array

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Returns coefficients of the trained model (for linear models like LogisticRegression)."""
        if hasattr(self.model, 'coef_'):
            # Для бинарной классификации coef_ - это 1D массив формы (n_features,)
            # Для мультиклассовой - это 2D (n_classes, n_features)
            # Мы возвращаем для положительного класса (индекс 0, если 2 класса, или весь массив для мультикласса)
            if len(self.model.classes_) == 2:
                 # Коэффициенты для класса 1 (положительный класс) против класса 0 (отрицательный класс)
                 # Знак указывает направление, величина указывает важность относительно других признаков
                 return self.model.coef_[0] 
            else:
                # Для мультиклассовой классификации возвращаем всю матрицу коэффициентов
                return self.model.coef_
        else:
            print("Model does not support feature importance via coefficients.")
            return None

    def get_feature_names(self) -> Optional[List[str]]:
        """Returns names of features (works for TF-IDF)."""
        if self.vectorizer_type == "tfidf" and hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            return self.tfidf_vectorizer.get_feature_names_out().tolist()
        elif self.vectorizer_type == "llm":
            embedding_dim = self.llm_vectorizer.encode_texts(["test"]).shape[1] # Получить размерность из фиктивного энкодирования
            return [f"embedding_dim_{i}" for i in range(embedding_dim)]
        else:
            return None

    def analyze_top_features(self, top_k: int = 10) -> List[Tuple[str, float]]:
        importance = self.get_feature_importance()
        names = self.get_feature_names()

        if importance is None or names is None:
            print("Cannot perform feature analysis: importance or names not available.")
            return []

        if len(importance) != len(names):
             print(f"Mismatch between importance ({len(importance)}) and names ({len(names)}) lengths.")
             return []

        # Создать пары и отсортировать по абсолютной важности
        feature_coef_pairs = list(zip(names, importance))
        # Сортировать по абсолютному значению коэффициента по убыванию
        sorted_pairs = sorted(feature_coef_pairs, key=lambda x: abs(x[1]), reverse=True)

        return sorted_pairs[:top_k]

    def save_model(self, filepath: str):
        # Сохраняет обученный векторизатор и модель.
        model_data = {
            'vectorizer_type': self.vectorizer_type,
            'model': self.model,
            'llm_vectorizer': self.llm_vectorizer if self.vectorizer_type == 'llm' else None,
            'tfidf_vectorizer': self.tfidf_vectorizer if self.vectorizer_type == 'tfidf' else None
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        # Загружает обученный векторизатор и модель.
        model_data = joblib.load(filepath)
        instance = cls(vectorizer_type=model_data['vectorizer_type'])
        instance.model = model_data['model']
        if instance.vectorizer_type == 'llm':
            instance.llm_vectorizer = model_data['llm_vectorizer']
        elif instance.vectorizer_type == 'tfidf':
            instance.tfidf_vectorizer = model_data['tfidf_vectorizer']
        print(f"Model loaded from {filepath}")
        return instance