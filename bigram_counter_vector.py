from tools import csv2dictlist
import collections


class BigramCounterVector():
    def __init__(self, fname, min_freq):
        """Read a ngram file and load ngram words that are over min_freq """
        # todo: Error check if fname has csv ext or not
        self.fname = fname
        self.min_freq = min_freq

        self.bigram_list = csv2dictlist(fname=fname)
        self.cvec_words = []
        for bigram_dict in self.bigram_list:
            if int(bigram_dict['frequency']) >= min_freq:
                bigram_tup = tuple(bigram_dict['2gram word'].split())
                self.cvec_words.append(bigram_tup)

        self.cvec = [0] * (len(self.cvec_words) + 1)

    def lines2cvec(self, lines, stopwords=['\n', '.']):
        cvec = [0] * (len(self.cvec_words) + 1)
        q = collections.deque(maxlen=2)

        words = lines.split(' ')
        for word in words:
            if word in stopwords:
                q.clear()
                continue

            q.append(word)
            if len(q) == 2:
                cvec[self.bigram2index(tuple(q))] += 1

        return cvec

    def bigram2index(self, bigram: tuple):
        """Convert one bigram to index of cvec"""
        if bigram in self.cvec_words:
            idx = self.cvec_words.index(bigram)
        else:
            idx = len(self.cvec_words)

        return idx

    def __len__(self):
        return len(self.cvec_words) + 1  # +1 for unknown words
