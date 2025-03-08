import pickle
from nltk.sentiment.util import mark_negation
from nltk.tokenize import word_tokenize
from ..training.extract_features import extract_feature
import os

def load_classifier(classifier_path):
    with open(classifier_path, "rb") as file:
        return pickle.load(file)

def classify_sentence(sentence):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_path = os.path.join(current_script_dir, "..", "models",
                                   "sentiment140_trained_model.pkl")
    classifier = load_classifier(classifier_path)

    words = mark_negation(word_tokenize(sentence.lower()))
    features = extract_feature(words)
    return classifier.classify(features)
