import math

class NaiveBayesClassifier:

    def __init__(self, data):
        self.data = data
        self.occurrences = dict()
        self.total_words = dict()
        self.probabilities = dict()
        self.total_samples = len(data)
        self.prior_probabilities = dict()
        self.vocab = set()

    def get_occurrences(self):
        for words, label in self.data:
            if label not in self.occurrences:
                self.occurrences[label] = dict()
            for word in words:
                if word in self.occurrences[label]:
                    self.occurrences[label][word] += 1
                else:
                    self.occurrences[label][word] = 1
                self.vocab.add(word)

    def get_total_words(self):
        for label, occurrences in self.occurrences.items():
            self.total_words[label] = sum(occurrences.values())

    def get_word_probabilities(self):
        vocab_size = len(self.vocab)
        for label, value in self.occurrences.items():
            self.probabilities[label] = dict()
            total_label_words = self.total_words[label]
            for word, occurrences in value.items():
                self.probabilities[label][word] = (occurrences + 1) / (total_label_words + vocab_size)

    def get_prior_prob(self):
        for label in self.occurrences:
            self.prior_probabilities[label] = sum(1 for _, l in self.data if l == label) / self.total_samples

    def train(self):
        self.get_occurrences()
        self.get_total_words()
        self.get_word_probabilities()
        self.get_prior_prob()

    def classify(self, sentence):
        likelihoods = dict()

        for label in self.probabilities:
            likelihoods[label] = math.log(self.prior_probabilities[label])
            for word in sentence:
                if word in self.probabilities[label]:
                    likelihoods[label] += math.log(self.probabilities[label][word])
                else:
                    likelihoods[label] += math.log(1 / self.total_words[label] + len(self.vocab))
        total_likelihood = sum(math.exp(lklhood) for lklhood in likelihoods.values())

        probabilities = {label: math.exp(lklhood) / total_likelihood for label, lklhood in likelihoods.items()}

        return probabilities