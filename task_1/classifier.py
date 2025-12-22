# -*- coding: utf-8 -*-

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import numpy as np
import torch

class ZeroShotClassifier:
    def __init__(self, model_name: str = "cointegrated/rubert-base-cased-nli-threeway"):
        device = -1  # По умолчанию CPU
        try:
            import torch
            if torch.cuda.is_available():
                device = 0
                print("Using GPU for ZeroShotClassifier.")
            else:
                print("GPU not available, using CPU for ZeroShotClassifier.")
        except ImportError:
            print("PyTorch not installed, using CPU for ZeroShotClassifier.")

        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device
            )
        except Exception as e:
            print(f"Model loading error {model_name}: {e}")
            raise
    
    def predict(self, texts: List[str], candidate_labels: List[str] = ["fake", "real"]) -> List[Dict]:
        results = []
        for text in texts:
            result = self.classifier(text, candidate_labels)
            results.append({
                "text": text,
                "predicted_label": result["labels"][0],
                "confidence": result["scores"][0],
                "all_scores": dict(zip(result["labels"], result["scores"]))
            })
        return results

class FewShotClassifier:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
                 
            self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"Error loading generative model {model_name} for FewShotClassifier: {e}")
            raise

    def predict(self, texts: List[str], examples: List[Dict] = None) -> List[Dict]:
        if examples is None:
            examples = [
                {"text": "The stock market reached an all-time high today after positive economic reports.", "label": "real"},
                {"text": "Aliens landed in Times Square and announced they come in peace.", "label": "fake"},
                {"text": "Scientists have found a cure for cancer using common household ingredients.", "label": "fake"},
                {"text": "The government announced new measures to combat climate change.", "label": "real"}
            ]

        results = []
        for text in texts:
            # Сформировать промпт
            prompt_parts = []
            for ex in examples:
                prompt_parts.append(f"News: {ex['text']}\nLabel: {ex['label'].upper()}\n---")
            
            prompt_parts.append(f"News: {text}\nLabel: ")
            prompt = "\n".join(prompt_parts)

            # Сгенерировать ответ
            response = self.generator(
                prompt,
                max_new_tokens=10,
                num_return_sequences=1,
                truncation=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            
            generated_text = response[0]['generated_text'][len(prompt):].strip()
            
            predicted_label_lower = generated_text.lower()
            if 'real' in predicted_label_lower:
                parsed_label = 'real'
            elif 'fake' in predicted_label_lower:
                parsed_label = 'fake'
            else:
                print(f"Warning: Unclear prediction for '{text[:50]}...': '{generated_text}'")
                parsed_label = 'fake' # Резервный вариант по умолчанию

            results.append({
                "text": text,
                "predicted_label": parsed_label,
                "raw_generation": generated_text
            })

        return results

# Тестирование
if __name__ == "__main__":
    print("--- Testing ZeroShotClassifier ---")
    classifier_zs = ZeroShotClassifier()
    test_texts = [
        "Scientists discover new form of life in deep oceans.",
        "Government announces sudden exit from country."
    ]
    predictions_zs = classifier_zs.predict(test_texts)
    for pred in predictions_zs:
        print(f"Text: {pred['text'][:50]}...")
        print(f"Prediction: {pred['predicted_label']} ({pred['confidence']:.2f})")
        print("---")