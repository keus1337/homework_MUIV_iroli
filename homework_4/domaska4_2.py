import pandas as pd
from natasha import Doc, MorphVocab, NewsNERTagger, NewsEmbedding, Segmenter
import spacy
from collections import defaultdict

# === Настройка инструментов ===
segmenter = Segmenter()
emb = NewsEmbedding()
morph_vocab = MorphVocab()
ner_tagger = NewsNERTagger(emb)
nlp = spacy.load("ru_core_news_sm")  # SpaCy модель для русского языка

# === Загрузка данных ===
data = pd.read_csv('generated_texts_with_topics.csv')
texts = data['generated_text'].tolist()
topics = data['LDA_Topic']  # Или другой столбец, если вы хотите использовать NMF или LSA

# === Функция для извлечения именованных сущностей Natasha ===
def extract_named_entities_natasha(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    entities = {
        'locations': [],
        'persons': [],
        'organizations': []
    }

    for span in doc.spans:
        span.normalize(morph_vocab)
        if span.type == 'LOC':
            entities['locations'].append(span.normal)
        elif span.type == 'PER':
            entities['persons'].append(span.normal)
        elif span.type == 'ORG':
            entities['organizations'].append(span.normal)

    return entities

# === Функция для извлечения именованных сущностей SpaCy ===
def extract_named_entities_spacy(text):
    doc = nlp(text)
    entities = {
        'locations': [],
        'persons': [],
        'organizations': []
    }
    for ent in doc.ents:
        if ent.label_ == 'LOC':
            entities['locations'].append(ent.text)
        elif ent.label_ == 'PER':
            entities['persons'].append(ent.text)
        elif ent.label_ == 'ORG':
            entities['organizations'].append(ent.text)
    return entities

# === Группировка сущностей по темам ===
entities_by_topic = defaultdict(lambda: {
    'locations': [],
    'persons': [],
    'organizations': []
})

for text, topic in zip(texts, topics):
    entities_natasha = extract_named_entities_natasha(text)
    entities_spacy = extract_named_entities_spacy(text)

    entities_by_topic[topic]['locations'].extend(entities_natasha['locations'] + entities_spacy['locations'])
    entities_by_topic[topic]['persons'].extend(entities_natasha['persons'] + entities_spacy['persons'])
    entities_by_topic[topic]['organizations'].extend(entities_natasha['organizations'] + entities_spacy['organizations'])

# Удаление дубликатов
for topic, entities in entities_by_topic.items():
    entities_by_topic[topic]['locations'] = list(set(entities['locations']))
    entities_by_topic[topic]['persons'] = list(set(entities['persons']))
    entities_by_topic[topic]['organizations'] = list(set(entities['organizations']))

# === Сохранение результатов ===
with open('named_entities_by_topic_combined.txt', 'w', encoding='utf-8') as file:
    for topic, entities in entities_by_topic.items():
        file.write(f"\n=== Тема {topic} ===\n")
        file.write(f"Географические названия: {', '.join(entities['locations'])}\n")
        file.write(f"Персоны: {', '.join(entities['persons'])}\n")
        file.write(f"Организации: {', '.join(entities['organizations'])}\n")

print("Результаты извлечения именованных сущностей сохранены в 'named_entities_by_topic_combined.txt'.")

# === Отладка: Проверка результатов для первых 10 текстов ===
with open('debug_entities.txt', 'w', encoding='utf-8') as debug_file:
    for i, text in enumerate(texts[:10]):
        entities_natasha = extract_named_entities_natasha(text)
        entities_spacy = extract_named_entities_spacy(text)
        debug_file.write(f"Текст {i+1}:\n{text}\n")
        debug_file.write(f"Natasha: {entities_natasha}\n")
        debug_file.write(f"SpaCy: {entities_spacy}\n\n")

print("Результаты отладки сохранены в 'debug_entities.txt'.")
