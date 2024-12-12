import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from datetime import datetime

### БЛОК ФУНКЦИЙ

# парсер
def parse_lenta():
    base_url = "https://lenta.ru"
    response = requests.get(base_url)

    if response.status_code != 200:
        print(f"Ошибка подключения: {response.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []

    for article in soup.find_all('a', class_="card-mini"):
        try:
            title_element = article.find('span', class_="card-mini__title")
            title = title_element.text.strip() if title_element else "Заголовок отсутствует"

            link = base_url + article['href'] if article['href'].startswith('/') else article['href']

            raw_date = article.get('data-gtm-date')
            date = (
                datetime.strptime(raw_date, '%Y-%m-%d %H:%M:%S') 
                if raw_date 
                else datetime.now()
            )

            article_response = requests.get(link)
            if article_response.status_code != 200:
                print(f"Не удалось загрузить статью: {link}")
                continue

            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            content_element = article_soup.find('div', class_="topic-body__content")
            content = content_element.text.strip() if content_element else "Текст отсутствует"

            articles.append({
                "title": title,
                "link": link,
                "date": date,
                "content": content
            })

        except Exception as e:
            print(f"Ошибка обработки статьи: {e}")
            continue

    return pd.DataFrame(articles)

# очистка текста
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.lower().strip()

# визуализация
def analyze_and_visualize(df):
    df['clean_content'] = df['content'].apply(lambda x: clean_text(str(x)))

    df = df.drop_duplicates(subset=['clean_content'])

    df = df[df['clean_content'].str.len() > 0]

    all_words = ' '.join(df['clean_content']).split()
    word_counts = Counter(all_words)

    stop_words = ['и', 'в', 'на', 'с', 'что', 'как', 'по', 'это', 'из', 'а', 'о', 'за', 'к', 'для', 'от', 'так']
    filtered_word_counts = {word: count for word, count in word_counts.items() if word not in stop_words}

    top_words = Counter(filtered_word_counts).most_common(10)
    print("Топ-10 наиболее частых слов:", top_words)

    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='skyblue')
    plt.title("Топ-10 наиболее частых слов")
    plt.xlabel("Слова")
    plt.ylabel("Частота")
    plt.xticks(rotation=45)
    plt.show()

    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    daily_counts = df['date_only'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(daily_counts.index, daily_counts.values, marker='o', color='green')
    plt.title("Распределение количества новостей по дням")
    plt.xlabel("Дата")
    plt.ylabel("Количество новостей")
    plt.grid(True)
    plt.show()

    return df

#### КОД
if __name__ == "__main__":
    print("Парсинг данных с Lenta.ru...")
    df = parse_lenta()

    if not df.empty:
        print("Парсинг завершён. Начинаем анализ данных.")
        cleaned_df = analyze_and_visualize(df)

        cleaned_df.to_csv("lenta_dataset_cleaned.csv", index=False, encoding='utf-8')
        print("Очищенные данные сохранены в lenta_dataset_cleaned.csv")
    else:
        print("Не удалось собрать данные.")
