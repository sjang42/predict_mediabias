from torch.utils.data import Dataset
from bigram_counter_vector import BigramCounterVector
from tools import csv2dictlist
import torch


class NewsDataset(Dataset):
    def __init__(self,
                 fname_parties,
                 fname_ngram,
                 ngram_min_freq=3,
                 train_ratio=0.8,
                 train=True):
        # TODO
        # 1. shuffle
        # 2. divide as train and test
        all_dict = csv2dictlist(fname_parties)

        self.data = []
        self.label = []
        for dic in all_dict:
            clean_text = dic['clean_text']
            press = dic['press']

            if clean_text == '':
                continue

            self.data.append(clean_text)
            self.label.append(0 if press == '더불어민주당' else 1)

        self.bigram_cvec = BigramCounterVector(
            fname_ngram, min_freq=ngram_min_freq)

    def __getitem__(self, index):
        # Convert to count vector
        data = self.bigram_cvec.lines2cvec(self.data[index])

        # Conver to Tensor
        data = torch.Tensor(data)
        label = torch.LongTensor([self.label[index]])

        return data, label

    def __len__(self):
        return len(self.data)

    def get_len_cvec(self):
        return len(self.bigram_cvec)
