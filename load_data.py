import re
import html

from tweet import Tweet
from nltk.corpus import stopwords


def remove_stopwords(string):
    new_stopwords = stopwords.words('english')
    new_stopwords.append("can't")
    split_string = \
        [word for word in string.split()
         if word not in new_stopwords]

    return " ".join(split_string)


def clean_string(string):
    string = remove_stopwords(string)
    string = html.unescape(string)
    string = string.replace("\\n", " ")
    string = string.replace("_NEG", "")
    string = string.replace("_NEGFIRST", "")
    string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string)  # removing any twitter handle mentions
    string = re.sub(r"\*", "", string)
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"\.", "", string)
    string = re.sub(r"-", "", string)
    string = re.sub(r"@", "", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"\s{2,}", " ", string)

    return remove_stopwords(string.strip().lower())


def read_train_data(train_file_path):
    train_list = list()
    with open(train_file_path, 'r') as input_file:
        for line in input_file:
            line = line.strip()
            array = line.split('\t')
            train_list.append(Tweet(array[0], clean_string(array[1]), array[2]))
    return train_list

