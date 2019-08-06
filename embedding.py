import gensim
import numpy as np

from nltk import word_tokenize

word_vector_path = 'C:/Users/jiwun/Desktop/word_vec'
wv_model_path = word_vector_path + '/GoogleNews-vectors-negative300.bin.gz'

wv_model = gensim.models.KeyedVectors.load_word2vec_format(wv_model_path, binary=True, unicode_errors='ignore')
w2v_dimensions = len(wv_model['word'])
# print(w2v_dimensions)


def get_word2vec_embedding(word, model, dimensions):
    vec_rep = np.zeros(dimensions)
    if word in model:
        vec_rep = model[word]

    return vec_rep


def build_word_embedding(tweets):
    max_tweet_length = -1
    word_embedding_dict = dict()

    for tweet in tweets:
        tokens = word_tokenize(tweet)

        if len(tokens) > max_tweet_length:
            max_tweet_length = len(tokens)

        for token in tokens:
            if token not in word_embedding_dict:
                word_embedding_dict[token] = get_word2vec_embedding(token, wv_model, w2v_dimensions)

    return word_embedding_dict, max_tweet_length


def vectorize_tweets(tweets, word_embedding_dict):
    train_vectors = list()
    for tweet in tweets:
        tokens = word_tokenize(tweet)

        train_vector = list()
        for token in tokens:
            train_vector.append(word_embedding_dict[token])

        train_vector = np.asarray(train_vector)
        train_vectors.append(train_vector)

    return np.asarray(train_vectors)


# sentence = ['Follow up. Follow through. Be . #success', 'I am a boy you are a girl']
#
# dict = vectorize_tweets(sentence)
# print(dict)