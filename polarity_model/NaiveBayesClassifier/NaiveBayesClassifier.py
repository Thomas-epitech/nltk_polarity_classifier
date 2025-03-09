class NaiveBayesClassifier:

    def __init__(self, data):
        self.data = data
        self.occurrences = dict()
        self.total_words = dict()
        self.probabilities = dict()

    def get_occurrences(self):
        for sample in self.data:
            key = str(sample[1])
            if key not in self.occurrences.keys():
                self.occurrences[key] = dict()
            for word in sample[0]:
                if word in self.occurrences[key].keys():
                    self.occurrences[key][word] += 1
                else:
                    self.occurrences[key][word] = 1

    def get_total_words(self):
        for key, occurrences in self.occurrences.items():
            self.total_words[key] = 0
            for occurrence in occurrences.values():
                self.total_words[key] += occurrence

    def get_word_probabilities(self):
        for key, value in self.occurrences.items():
            self.probabilities[key] = dict()
            for word, occurrences in value.items():
                self.probabilities[key][word] = occurrences / self.total_words[key]

    def train(self):
        self.get_occurrences()
        self.get_total_words()
        self.get_word_probabilities()

    def classify(self, sentence):
        likelihoods = dict()

        for key, value in self.probabilities.items():
            likelihoods[key] = 1
            for word in sentence:
                try:
                    likelihoods[key] *= self.probabilities[key][word]
                except KeyError:
                    likelihoods[key] *= 1 / self.total_words[key]
        return likelihoods