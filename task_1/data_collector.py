# -*- coding: utf-8 -*-

import json
import requests
import os
from typing import List, Dict
from datetime import datetime
import feedparser
import time

def collect_real_news_from_rss(rss_url: str, limit: int = 5) -> List[Dict]:
    try:
        print(f"Fetching news from RSS: {rss_url}")
        feed = feedparser.parse(rss_url)
        
        news_items = []
        for entry in feed.entries[:limit]:
            title = getattr(entry, 'title', '')
            summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
            # Очистить HTML-теги, если они присутствуют в сводке
            import re
            summary_clean = re.sub('<[^<]+?>', '', summary)
            
            news_items.append({
                "title": title.strip(),
                "content": summary_clean.strip()
            })
            
        print(f"Fetched {len(news_items)} news items from RSS.")
        return news_items
    except Exception as e:
        print(f"Error fetching RSS feed {rss_url}: {e}")
        return []

def collect_real_news() -> List[Dict]:
    rss_url = "http://feeds.bbci.co.uk/news/rss.xml" # BBC News
    
    # Получить новости из RSS
    real_news = collect_real_news_from_rss(rss_url, limit=5) 
    
    # Резервный вариант, если RSS не срабатывает
    if not real_news:
        print("RSS collection failed, using stub data.")
        real_news = [
            {"title": "Russia launches new communication satellite", 
             "content": "Today at 12:00 MSK, a carrier rocket successfully launched with a new communication satellite."},
            {"title": "Scientists discover new species in Amazon rainforest", 
             "content": "A team of international researchers has identified a previously unknown species of frog with unique vocalizations."},
            {"title": "Global tech summit announces breakthrough in quantum computing", 
             "content": "Leading companies presented advances in error correction for quantum processors."},
            {"title": "Renewable energy project reaches major milestone", 
             "content": "A new solar farm installation has become operational, providing clean power for over 100,000 homes."},
            {"title": "Historical artifact discovered during archaeological dig", 
             "content": "Archaeologists unearthed a collection of ancient coins dating back to the 4th century."}
        ]
    return real_news

def generate_fake_news_with_llm(model_name: str, count: int) -> List[str]:
    fake_samples = []
    for i in range(count):
        # Смоделировать различные типы фейковых новостей
        topics = ["Political", "Science", "Technology", "Health", "Environment"]
        topic = topics[i % len(topics)]
        fake_samples.append(
            f"{topic} Hoax {i+1}: Experts reveal shocking truth about {topic.lower()} that governments don't want you to know!"
        )
    return fake_samples

def create_dataset(output_path: str):
    # Создает набор данных с реальными и фальшивыми новостями
    print("=== Collecting real news ===")
    real_news = collect_real_news()
    
    fake_models = ["gpt", "llama", "mistral", "yi"]
    all_fake = []
    for model in fake_models:
        # Распределить генерацию фейков более равномерно
        count_for_model = len(real_news) // len(fake_models)
        # Обработать остаток, если len(real_news) не делится нацело
        if fake_models.index(model) < len(real_news) % len(fake_models):
            count_for_model += 1
            
        fake_batch = generate_fake_news_with_llm(model, count_for_model)
        all_fake.extend(fake_batch)
    
    # Объединить в один набор данных
    data = []
    for item in real_news:
        full_text = f"{item.get('title', '')}. {item.get('content', '')}".strip()
        if full_text:
            data.append({"text": full_text, "label": 1})  # 1 = real
    
    for item in all_fake:
        if item.strip():
            data.append({"text": item.strip(), "label": 0})  # 0 = fake
    
    # Убедиться, что директория существует перед записью
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Сохранить
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_path}, records: {len(data)}")
    print(f"  - Real news: {sum(1 for d in data if d['label'] == 1)}")
    print(f"  - Fake news: {sum(1 for d in data if d['label'] == 0)}")


if __name__ == "__main__":
    create_dataset("dataset/train_data.json")