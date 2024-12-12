import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import nltk

# Загрузка данных
nltk.download('punkt')
nltk.download('stopwords')
data = pd.read_csv('generated_texts.csv')
texts = data['generated_text'].dropna().tolist()

# Очистка текста
def preprocess_text(text):
    text = re.sub(r'[^а-яА-Я\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

processed_texts = [preprocess_text(text) for text in texts]

# Векторизация текста
vectorizer = CountVectorizer(max_features=1000)
x_counts = vectorizer.fit_transform(processed_texts)

# === Тематическое моделирование ===
# 1. LDA
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda_model.fit_transform(x_counts)

# 2. LSA (Truncated SVD)
lsa_model = TruncatedSVD(n_components=5, random_state=42)
lsa_topics = lsa_model.fit_transform(x_counts)

# 3. NMF
nmf_model = NMF(n_components=5, random_state=42)
nmf_topics = nmf_model.fit_transform(x_counts)

# Функция для отображения тем
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics

no_top_words = 10
feature_names = vectorizer.get_feature_names_out()

lda_topics_display = display_topics(lda_model, feature_names, no_top_words)
lsa_topics_display = display_topics(lsa_model, feature_names, no_top_words)
nmf_topics_display = display_topics(nmf_model, feature_names, no_top_words)

# Отображение тем
print("\nLDA Темы:")
for topic, words in lda_topics_display.items():
    print(f"{topic}: {', '.join(words)}")

print("\nLSA Темы:")
for topic, words in lsa_topics_display.items():
    print(f"{topic}: {', '.join(words)}")

print("\nNMF Темы:")
for topic, words in nmf_topics_display.items():
    print(f"{topic}: {', '.join(words)}")

# Определение темы для каждого текста
lda_assigned_topics = lda_topics.argmax(axis=1)
nmf_assigned_topics = nmf_topics.argmax(axis=1)
lsa_assigned_topics = lsa_topics.argmax(axis=1)

# Добавление тем в DataFrame
data['LDA_Topic'] = lda_assigned_topics
data['NMF_Topic'] = nmf_assigned_topics
data['LSA_Topic'] = lsa_assigned_topics

# Сохранение результатов
data.to_csv('generated_texts_with_topics.csv', index=False)
print("Результаты кластеризации сохранены в 'generated_texts_with_topics.csv'.")
