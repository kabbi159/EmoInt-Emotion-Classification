from load_data import read_train_data
from tweet import emotion2id
from embedding import vectorize_tweets, build_word_embedding

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D

from sklearn.model_selection import train_test_split

import numpy as np
import random
SEED = 448
random.seed(SEED)


emoint_path = 'C:/Users/jiwunghyun/Desktop/EmoInt'
train_file_path = 'C:/Users/jiwun/Desktop/train/'
emotion_path = ['anger.txt', 'fear.txt', 'joy.txt', 'sadness.txt']

tweet_train = list()
emotion_train = list()

# read data + preprocess text
for emotion in emotion_path:
    train_tweets_object = read_train_data(train_file_path + emotion)

    for tweet in train_tweets_object:
        tweet_train.append(tweet.text)
        emotion_train.append(emotion2id(tweet.emotion))


# Shuffle data
combine = list(zip(tweet_train, emotion_train))
random.shuffle(combine)

tweet_train_shuffle, emotion_train_shuffle = zip(*combine)

# Split data (Train : Test = 0.8 : 0.2) - 챌린지 데이터라서 train 데이터밖에 구하지 못하였음.
# 따라, train 데이터를 split 하여 사용
x_train, x_test, y_train, y_test = train_test_split(tweet_train_shuffle, emotion_train_shuffle, test_size=0.2)

# y는 리스트 형태임 -> numpy array 로 변환
y_train = np.array(y_train)
y_test = np.array(y_test)

# text -> vector
word_embedding_dict, max_tweet_length = build_word_embedding(tweet_train)
x_train = vectorize_tweets(x_train, word_embedding_dict)
x_train = sequence.pad_sequences(x_train, maxlen=max_tweet_length, padding="post", truncating="post",
                                 dtype='float64')
x_test = vectorize_tweets(x_test, word_embedding_dict)
x_test = sequence.pad_sequences(x_test, maxlen=max_tweet_length, padding="post", truncating="post",
                                dtype='float64')

print(max_tweet_length)  # 80


# Deep Learning Model + hyperparameter
batch_size = 4
conv_kernel = 5

"""
Model: Conv1D - MaxPool1D - Dense(1000) - Dense(4)
"""

model = Sequential()
model.add(Conv1D(300, conv_kernel, activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPool1D())
model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=10)


"""
Test on test data
"""

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Loss:", loss_and_metrics[0], "Test Accuracy:", loss_and_metrics[1])  # 1.09, 0.806
