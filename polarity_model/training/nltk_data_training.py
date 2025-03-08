from nltk import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.sentiment.util import *
from extract_features import extract_feature
import random
import pickle

def get_train_data():
    positive_reviews = [mark_negation(list(movie_reviews.words(fileid)))
                        for fileid in movie_reviews.fileids('pos')]
    negative_reviews = [mark_negation(list(movie_reviews.words(fileid)))
                        for fileid in movie_reviews.fileids('neg')]
    featured_positive_reviews = [(extract_feature(pos_review), 'pos')
                                for pos_review in positive_reviews]
    featured_negative_reviews = [(extract_feature(neg_review), 'neg')
                                for neg_review in negative_reviews]

    train_data = featured_positive_reviews + featured_negative_reviews
    random.shuffle(train_data)
    return train_data

def train_classifier():
    train_data = get_train_data()
    print(len(train_data))
    classifier = NaiveBayesClassifier.train(train_data)

    with open("../models/nltk_data_trained_model.pkl", "wb") as file:
        pickle.dump(classifier, file)

train_classifier()