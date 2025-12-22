# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class LLMVectorizer:
    # Векторизация текста с использованием предобученной модели
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Model loading error {model_name}: {e}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        # Преобразует список текстов в матрицу эмбеддингов
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts)
        return embeddings

# Тестирование
if __name__ == "__main__":
    vectorizer = LLMVectorizer()
    test_texts = [
        "This is a real news about a scientific discovery.",
        "This is a fake news about a fictional event."
    ]
    vectors = vectorizer.encode_texts(test_texts)
    print(f"Embedding shape: {vectors.shape}")