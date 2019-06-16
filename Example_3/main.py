#! /usr/bin/env python3

# Туториал из урока https://www.youtube.com/watch?v=CDpbJIbDhys&t=0s
# [Скачать корпус твиттов](http://study.mokoron.com);
# [Ю. В. Рубцова. Построение корпуса текстов для настройки тонового классификатора // Программные продукты и системы, 2015, №1(109), –С.72-78](http://www.swsys.ru/index.php?page=article&id=3962&lang=);"


import re
import collections
import os
import json
import random
import tflearn
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer


POSITIVE_TWEETS_CSV = 'positive.csv'
NEGATIVE_TWEETS_CSV = 'negative.csv'
VOCAB_FILE = 'stem_vocab.json'
STEM_COUNT_FILE = 'stem_count.json'

TWEETS_COL_NUMBER = 3

TRAIN_TWEETS_COUNT = 5000

#VOCAB_SIZE = 5000
VOCAB_SIZE = 3000


def get_stem(stem_cache, regex, stemer, token):
    stem = stem_cache.get(token, None)
    if stem:
        return stem
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem

def count_unique_tokens_in_tweets(tweets, tokenizer, stem_cache, regex, stemer, stem_count):
    for _, tweet_series in tweets.iterrows():
        tweet = tweet_series[3]
        tokens = tokenizer.tokenize(tweet)
        for token in tokens:
            stem = get_stem(stem_cache, regex, stemer, token)
            stem_count[stem] += 1

def tweet_to_vector(tweet, tokenizer, token_2_idx, stem_cache, regex, stemer, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(tweet):
        stem = get_stem(stem_cache, regex, stemer, token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif show_unknowns:
            print("Unknown token: {}".format(token))
    return vector

def build_model(learning_rate=0.1):
    tf.reset_default_graph()
    
    net = tflearn.input_data([None, VOCAB_SIZE])
    net = tflearn.fully_connected(net, 1024, activation='ReLU')
    #net = tflearn.fully_connected(net, 256, activation='ReLU')
    #net = tflearn.fully_connected(net, 128, activation='ReLU')
    net = tflearn.fully_connected(net, 64, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net, 
        optimizer='sgd', 
        learning_rate=learning_rate, 
        loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model



def test_tweet(tweet, model, tokenizer, token_2_idx, stem_cache, regex, stemer):
    tweet_vector = tweet_to_vector(tweet, tokenizer, token_2_idx, stem_cache, regex, stemer, True)
    positive_prob = model.predict([tweet_vector])[0][1]
    print('Original tweet: {}'.format(tweet))
    print('P(positive) = {:.5f}. Result: '.format(positive_prob), 'Positive' if positive_prob > 0.5 else 'Negative')


def test_tweet_number(idx, tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer):
    if idx < len(tweets):
        test_tweet(tweets[idx], model, tokenizer, token_2_idx, stem_cache, regex, stemer)
    else:
        print("Out of range")


def main():

    # Загружаем непосредственно сообщения
    negative_tweets = pd.read_csv(POSITIVE_TWEETS_CSV, header=None, delimiter=";")[[TWEETS_COL_NUMBER]]
    positive_tweets = pd.read_csv(NEGATIVE_TWEETS_CSV, header=None, delimiter=";")[[TWEETS_COL_NUMBER]]

    # ограничение количества твитов для анализа
    negative_tweets = negative_tweets[0:min(TRAIN_TWEETS_COUNT, len(negative_tweets))]
    positive_tweets = positive_tweets[0:min(TRAIN_TWEETS_COUNT, len(positive_tweets))]

    # Создаем стемер (стем - основа слова)
    stemer = RussianStemmer()
    # Регулярка для фильтрации
    regex = re.compile('[^а-яА-Я ]')
    # Кеш повторяющихся стемов
    stem_cache = {}

    tokenizer = TweetTokenizer()


    if os.path.exists(VOCAB_FILE) == False:
        stem_count = collections.Counter()

        # Получаем уникальные токены в твитах
        count_unique_tokens_in_tweets(negative_tweets, tokenizer, stem_cache, regex, stemer, stem_count)
        count_unique_tokens_in_tweets(positive_tweets, tokenizer, stem_cache, regex, stemer, stem_count)

        print("Total unique stems found: ", len(stem_count))

        # TODO: Надо ли конвертировать в словарь?
        # stem_count = dict(stem_count)

        # Создаем словарь
        vocab = sorted(stem_count, key=stem_count.get, reverse=True)[:VOCAB_SIZE]
        print(vocab[:100])

        with open(VOCAB_FILE, "w") as f:
            json.dump(vocab, f)
        with open(STEM_COUNT_FILE, "w") as f:
            json.dump(stem_count, f)
    else:
        with open(VOCAB_FILE, "r") as f:
            vocab = json.load(f)
        with open(STEM_COUNT_FILE, "r") as f:
            stem_count = json.load(f)


    idx = 2
    print("stem: {}, count: {}".format(vocab[idx], stem_count.get(vocab[idx])))

    token_2_idx = {vocab[i]: i for i in range(VOCAB_SIZE)}
    len(token_2_idx)

    print(token_2_idx['сказа'])

    tweet = negative_tweets.iloc[1][3]
    print("tweet: {}".format(tweet))
    print("vector: {}".format(tweet_to_vector(tweet, tokenizer, token_2_idx, stem_cache, regex, stemer)[:10]))
    print(vocab[5])

    total_size = len(negative_tweets) + len(positive_tweets)
    tweet_vectors = np.zeros((total_size, VOCAB_SIZE), dtype=np.int_)
    tweets = []
    for ii, (_, tweet) in enumerate(negative_tweets.iterrows()):
        tweets.append(tweet[TWEETS_COL_NUMBER])
        tweet_vectors[ii] = tweet_to_vector(tweet[TWEETS_COL_NUMBER], tokenizer, token_2_idx, stem_cache, regex, stemer)
    for ii, (_, tweet) in enumerate(positive_tweets.iterrows()):
        tweets.append(tweet[TWEETS_COL_NUMBER])
        tweet_vectors[ii + len(negative_tweets)] = tweet_to_vector(tweet[TWEETS_COL_NUMBER], tokenizer, token_2_idx, stem_cache, regex, stemer)

    labels = np.append(np.zeros(len(negative_tweets), dtype=np.int_), np.ones(len(positive_tweets), dtype=np.int_))

    # labels[:10]
    # labels[-10:]

    X = tweet_vectors
    y = to_categorical(labels, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print(y_test[:10])

    model = build_model(learning_rate=0.6)

    model.fit(
        X_train,
        y_train,
        validation_set=0.1,
        show_metric=True,
        snapshot_epoch=False,
        batch_size=128,
        n_epoch=50)

    predictions = (np.array(model.predict(X_test))[:, 0] >= 0.5).astype(np.int_)
    accuracy = np.mean(predictions == y_test[:, 0], axis=0)
    print("Accuracy: ", accuracy)

    test_tweet_number(random.randint(0, len(tweets)), tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer)
    test_tweet_number(random.randint(0, len(tweets)), tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer)
    test_tweet_number(random.randint(0, len(tweets)), tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer)
    test_tweet_number(random.randint(0, len(tweets)), tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer)
    test_tweet_number(random.randint(0, len(tweets)), tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer)
    test_tweet_number(random.randint(0, len(tweets)), tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer)
    test_tweet_number(random.randint(0, len(tweets)), tweets, model, tokenizer, token_2_idx, stem_cache, regex, stemer)

    tweets_for_testing = [
        "Меня оштрафовали по дороге домой",
        "Веселый был день... Даже не знаю что и сказать"
    ]
    for tweet in tweets_for_testing:
        test_tweet(tweet, model, tokenizer, token_2_idx, stem_cache, regex, stemer)
        print("---------")

if __name__ == '__main__':
    main()
