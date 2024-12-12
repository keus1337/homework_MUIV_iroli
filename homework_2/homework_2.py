import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

file_path = 'anek.txt'  # Не придумал файла лучше

with open(file_path, 'r', encoding='utf-8') as file:
    texts = file.readlines()

def preprocess(text):
    stop_words = set(stopwords.words('russian'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

preprocessed_texts = [preprocess(text) for text in texts]

tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(text) for text in preprocessed_texts])

print("TF-IDF матрица:")
print(tfidf_matrix.shape)

word2vec_model = Word2Vec(sentences=preprocessed_texts, vector_size=100, window=5, min_count=1, workers=4)

def get_word2vec_embedding(text, model):
    tokens = text
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

word2vec_embeddings = np.array([get_word2vec_embedding(text, word2vec_model) for text in preprocessed_texts])

inertia = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(word2vec_embeddings)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 6), inertia, marker='o')
plt.xlabel('Количество кластеров')
plt.ylabel('Инэрция (SSE)')
plt.title('Метод локтя для выбора оптимального k')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(word2vec_embeddings)

labels = kmeans.labels_

for i, text in enumerate(texts):
    print(f"Text: {text.strip()}")
    print(f"Cluster: {labels[i]}")
