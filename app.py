import sys
from polarity_model.using.nltk_data_trained import classify_sentence

def run():
    nb_args = len(sys.argv)

    if nb_args > 2:
        raise TypeError("Program must not exceed 2 parameters.")

    if nb_args == 1:
        sentence = input("Sentence: ")
    else:
        sentence = sys.argv[1]
    print(classify_sentence(sentence))