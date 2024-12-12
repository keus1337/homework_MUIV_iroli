import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Загрузка данных
data = pd.read_csv('lenta_dataset_cleaned.csv')

# 2. Создание целевой переменной
def assign_category(text):
    if 'украин' in text or 'киев' in text or 'запорож' in text:
        return 'Ukraine'
    elif 'росси' in text or 'москва' in text or 'путин' in text:
        return 'Russia'
    elif 'спорт' in text or 'матч' in text or 'футбол' in text:
        return 'Sports'
    else:
        return 'Other'

data['category'] = data['clean_content'].apply(assign_category)

# 3. Разделение на обучающую и тестовую выборки
X = data['clean_content']
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Преобразование текста в признаки (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Обучение классификатора (Наивный Байес)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Предсказание на тестовой выборке
y_pred = model.predict(X_test_tfidf)

# 7. Оценка качества модели
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Матрица смежности
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
