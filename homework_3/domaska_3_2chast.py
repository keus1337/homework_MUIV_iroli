import pandas as pd
import re
import random
from collections import defaultdict, Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, LSTM, Dense, Dropout
import numpy as np
import os

# === 1. Подготовка данных ===
# Загрузка данных
data = pd.read_csv('lenta_dataset_cleaned.csv')
corpus = ' '.join(data['clean_content'][:5000])  # Берём первые 5000 текстов и объединяем их

# Очистка текста
corpus = re.sub(r'\s+', ' ', corpus)
corpus = re.sub(r'[^а-яА-Я\s]', '', corpus.lower())

# === 2. Статистическая языковая модель (n-граммы) ===
def build_ngram_model(text, n=3):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    model = defaultdict(Counter)
    for *prefix, next_word in ngrams:
        model[tuple(prefix)][next_word] += 1
    return model

def generate_ngram_text(model, n=3, max_words=50, seed=None):
    if not seed:
        seed = random.choice(list(model.keys()))
    text = list(seed)
    for _ in range(max_words - len(seed)):
        prefix = tuple(text[-(n-1):])
        if prefix in model:
            next_word = random.choices(
                list(model[prefix].keys()), 
                weights=model[prefix].values()
            )[0]
            text.append(next_word)
        else:
            break
    return ' '.join(text)

# Создаём статистическую модель
ngram_model = build_ngram_model(corpus, n=3)

# Генерация текстов статистической моделью
print("\n=== Тексты, сгенерированные статистической моделью ===")
for i in range(10):
    seed = random.choice(list(ngram_model.keys()))
    print(f"Текст {i+1}: {generate_ngram_text(ngram_model, seed=seed)}")

# === 3. Подготовка данных для нейросетевых моделей ===
tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus])
total_words = len(tokenizer.word_index) + 1

# Создание последовательностей
input_sequences = []
for line in corpus.split('. '):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding последовательностей
max_sequence_len = 20  # Ограничиваем длину последовательности
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Разделение на X и y
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = to_categorical(y, num_classes=total_words)

# === 4. Создание и генерация с использованием моделей (SimpleRNN, GRU, LSTM) ===
# Семплирование для более разнообразного текста
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_rnn_text(model, tokenizer, seed_text, max_words=50, temperature=1.0):
    sequence = tokenizer.texts_to_sequences([seed_text])
    sequence = pad_sequences(sequence, maxlen=20, padding='pre')
    generated = seed_text
    for _ in range(max_words):
        pred = model.predict(sequence, verbose=0)[0]
        next_word_index = sample(pred, temperature)
        next_word = tokenizer.index_word.get(next_word_index, '')
        if not next_word:
            break
        generated += ' ' + next_word
        sequence = pad_sequences(tokenizer.texts_to_sequences([generated]), maxlen=20, padding='pre')
    return generated

# Функции для создания моделей
def create_rnn_model():
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))
    model.add(SimpleRNN(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model():
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))
    model.add(GRU(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model():
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Обучение моделей
simple_rnn_model = create_rnn_model()
simple_rnn_model.fit(X, y, epochs=10, batch_size=64, verbose=1)

gru_model = create_gru_model()
gru_model.fit(X, y, epochs=10, batch_size=64, verbose=1)

lstm_model = create_lstm_model()
lstm_model.fit(X, y, epochs=10, batch_size=64, verbose=1)

# Генерация текстов
print("\n=== Генерация текстов ===")
generated_texts = []

# Генерация для SimpleRNN
for i in range(10):
    seed_text = random.choice(corpus.split('.')).strip()[:20]
    generated_texts.append(f"SimpleRNN {i+1}: {generate_rnn_text(simple_rnn_model, tokenizer, seed_text, temperature=1.0)}")

# Генерация для GRU
for i in range(10):
    seed_text = random.choice(corpus.split('.')).strip()[:20]
    generated_texts.append(f"GRU {i+1}: {generate_rnn_text(gru_model, tokenizer, seed_text, temperature=1.0)}")

# Генерация для LSTM
for i in range(10):
    seed_text = random.choice(corpus.split('.')).strip()[:20]
    generated_texts.append(f"LSTM {i+1}: {generate_rnn_text(lstm_model, tokenizer, seed_text, temperature=1.0)}")

# === Сохранение текстов ===
if os.path.exists('generated_texts.csv'):
    os.remove('generated_texts.csv')

# Сохранение в CSV
df = pd.DataFrame({'generated_text': generated_texts})
df.to_csv('generated_texts.csv', index=False)
print("Тексты сохранены в файл 'generated_texts.csv'.")
