from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.sentiment.util import *
import random
import pickle

def extract_feature(words):
    stop_words = set(stopwords.words('english'))
    negated_stop_words = set({})
    for word in stop_words:
        negated_stop_words.add(word)
        negated_stop_words.add(word + "_NEG")

    filtered_words = [word for word in words if word not in negated_stop_words]
    return {word: True for word in filtered_words}

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
    classifier = NaiveBayesClassifier.train(train_data)

    with open("../models/nltk_data_trained_model.pkl", "wb") as file:
        pickle.dump(classifier, file)

"""
def classify_sentence(sentence):
    words = mark_negation(word_tokenize(sentence.lower()))
    features = extract_feature(words)
    print(features)
    return classifier.classify(features)

test_sentences = [
    "I’m not happy with my appearance.",
    "I’m frustrated with my progress.",
    "I’m worried about the future.",
    "I’m not satisfied with my work.",
    "I’m not happy with the direction my life is taking."
]

for s in test_sentences:
    print(f"Sentence: {s}\nSentiment: {classify_sentence(s)}\n")
"""
