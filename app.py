import sys
from polarity_model.using.classify_sentence import classify_sentence

def run():
    nb_args = len(sys.argv)

    if nb_args > 2:
        raise TypeError("Program must not exceed 2 parameters.")

    if nb_args == 1:
        sentence = input("Sentence: ")
    else:
        sentence = sys.argv[1]
    classification = classify_sentence(sentence)
    print("Your sentence is ", end='')
    if classification["0"] > classification["4"]:
        print("NEGATIVE")
    else:
        print("POSITIVE")
