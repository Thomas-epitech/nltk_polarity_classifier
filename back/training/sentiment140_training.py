import os
import csv
import sys
import pickle
import random
from nltk.tokenize import word_tokenize
from nltk.sentiment.util import mark_negation
from nltk.corpus import stopwords

training_dir = os.path.dirname(os.path.abspath(__file__))
polarity_model_dir = os.path.dirname(training_dir)

sys.path.append(os.path.join(polarity_model_dir, "NaiveBayesClassifier"))
sys.path.append(os.path.join(polarity_model_dir, "load_bar"))

from NaiveBayesClassifier import NaiveBayesClassifier
from loading import loading


nb_tweets = 1_600_000

def extract_tweets(file):
    extracted_tweets = []
    iteration = 0

    for row in file:
        extracted_tweets.append((row[5], int(row[0])))
        loading(iteration, nb_tweets, prefix="Extracting training data", suffix="Complete", loaded='▓', unloaded='▒', length=50, left_side='', right_side='')
        iteration += 1
    return extracted_tweets


def lowercase(tweets):
    lowercased_tweets = []
    iteration = 0

    for t in tweets:
        lowercased_tweets.append((t[0].lower(), t[1]))
        loading(iteration, nb_tweets, prefix="Lowercasing training data", suffix="Complete", loaded='▓', unloaded='▒', length=50, left_side='', right_side='')
        iteration += 1
    return lowercased_tweets

def tokenize(tweets):
    tokenized_tweets = []
    iteration = 0

    for t in tweets:
        tokenized_tweets.append((word_tokenize(t[0]), t[1]))
        loading(iteration, nb_tweets, prefix="Tokenizing training data", suffix="Complete", loaded='▓', unloaded='▒', length=50, left_side='', right_side='')
        iteration += 1
    return tokenized_tweets

def filter_words(tweets):
    filtered_tweets = []
    iteration = 0
    stop_words = set(stopwords.words('english'))

    for t in tweets:
        filtered = []
        for word in t[0]:
            if word not in stop_words:
                filtered.append(word)
        filtered_tweets.append((filtered, t[1]))
        loading(iteration, nb_tweets, prefix="Filtering words", suffix="Complete", loaded='▓', unloaded='▒', length=50, left_side='', right_side='')
        iteration += 1
    return filtered_tweets



def negate(tweets):
    tokenized_tweets = []
    iteration = 0

    for t in tweets:
        tokenized_tweets.append((mark_negation(t[0]), t[1]))
        loading(iteration, nb_tweets, prefix="Marking negation", suffix="Complete", loaded='▓', unloaded='▒', length=50, left_side='', right_side='')
        iteration += 1
    return tokenized_tweets

def extract_features(tweets):
    stop_words = set(stopwords.words('english'))
    negated_stop_words = set({})
    nb_stop_words = len(stop_words)
    featured_tweets = []
    iteration = 0

    for word in stop_words:
        negated_stop_words.add(word)
        negated_stop_words.add(word + "_NEG")
        loading(iteration, nb_stop_words, prefix="Fetching stop words", suffix="Complete", loaded='▓', unloaded='▒', length=50, left_side='', right_side='')
        iteration += 1

    iteration = 0
    for t in tweets:
        filtered_words = [word for word in t[0] if word not in negated_stop_words]
        featured_tweets.append(({word: True for word in filtered_words}, t[1]))
        loading(iteration, nb_tweets, prefix="Extracting features", suffix="Complete", loaded='▓', unloaded='▒', length=50, left_side='', right_side='')
        iteration += 1
    return featured_tweets



def get_train_data(path_to_data):

    with open(path_to_data, newline='', encoding='windows-1252') as data_file:
        file_read = csv.reader(data_file)
        return negate(filter_words(tokenize(lowercase(extract_tweets(file_read)))))

def train():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_script_dir, "data", "sentiment140.csv")
    model_path = os.path.join(current_script_dir, "..", "models", "sentiment140_trained_model.pkl")

    train_data = get_train_data(data_path)
    print("Shuffling data ...")
    random.shuffle(train_data)
    classifier = NaiveBayesClassifier(train_data)
    print("Training model...")
    classifier.train()
    with open(model_path, "wb") as file:
        pickle.dump(classifier, file)