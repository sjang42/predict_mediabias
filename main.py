from bigram_counter_vector import BigramCounterVector
import torch
from torch.utils.data import Dataset
from tools import csv2dictlist


class NewsDataset(Dataset):
    def __init__(self, fname_parties, fname_ngram, ngram_min_freq=3, train_ratio=0.8, train=True):
        # TODO
        # 1. shuffle
        # 2. divide as train and test
        all_dict = csv2dictlist(fname)

        self.data = []
        self.label = []
        for dic in all_dict:
            clean_text = dic['clean_text']
            press = dic['press']

            if clean_text == '':
                continue

            self.data.append(clean_text)
            self.label.append(-1 if press == '더불어민주당' else 1)

        self.bigram_cvec = BigramCounterVector(fname_ngram, min_freq=ngram_min_freq)

    def __getitem__(self, index):
        # Convert to count vector
        data = self.bigram_cvec.lines2cvec(self.data[index])

        # Conver to Tensor
        data = torch.Tensor(data)
        label = torch.Tensor([self.label[index]])

        return data, label

    def __len__(self):
        return len(self.data)


fname = 'data/train/parties_merged.csv'
ngram_path = 'ngrams/bigram_merged.csv'
news_dataset = NewsDataset(fname_parties=fname, fname_ngram=ngram_path, ngram_min_freq=4)

