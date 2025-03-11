import pickle
from nltk.sentiment.util import mark_negation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pprint import pprint
import os
import sys

training_dir = os.path.dirname(os.path.abspath(__file__))
polarity_model_dir = os.path.dirname(training_dir)

sys.path.append(os.path.join(polarity_model_dir, "NaiveBayesClassifier"))
sys.path.append(os.path.join(polarity_model_dir, "training"))

from sentiment140_training import train

def load_classifier(classifier_path):
    with open(classifier_path, "rb") as file:
        return pickle.load(file)

def filter(sentence):
    filtered = []
    stop_words = stopwords.words('english')
    negated_stop_words = [item for sublist in [[sw, (sw + "_NEG")] for sw in stop_words] for item in sublist]

    for word in sentence:
        if word not in negated_stop_words:
            filtered.append(word)
    return filtered

def classify_sentence(sentence):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_path = os.path.join(current_script_dir, "..", "models",
                                   "sentiment140_trained_model.pkl")
    if not os.path.isfile(classifier_path):
        train()
    classifier = load_classifier(classifier_path)

    words = mark_negation(filter(word_tokenize(sentence.lower())))
    return classifier.classify(words)