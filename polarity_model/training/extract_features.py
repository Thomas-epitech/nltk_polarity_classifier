from nltk.corpus import stopwords

def extract_feature(words):
    stop_words = set(stopwords.words('english'))
    negated_stop_words = set({})
    for word in stop_words:
        negated_stop_words.add(word)
        negated_stop_words.add(word + "_NEG")

    filtered_words = [word for word in words if word not in negated_stop_words]
    return {word: True for word in filtered_words}
