from nltk.corpus import stopwords
import numpy as np


def remove_stopwords(string):
    new_stopwords = stopwords.words('english')
    new_stopwords.append("can't") # stopwords do not include "can't"
    split_string = \
        [word for word in string.split()
         if word not in new_stopwords]

    return " ".join(split_string)


class Tweet(object):
    def __init__(self, id, text, emotion):
        self.id = id
        self.text = text
        self.emotion = emotion


def emotion2id(emotion):
    emotions = ['anger', 'fear', 'joy', 'sadness']
    nb_classes = len(emotions)
    for i in range(len(emotions)):
        if emotion == emotions[i]:
            return np.eye(nb_classes)[i]
