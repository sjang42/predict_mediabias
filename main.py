from bigram_counter_vector import BigramCounterVector
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tools import csv2dictlist
import torch.nn as nn


class NewsDataset(Dataset):
    def __init__(self, fname_parties, fname_ngram, ngram_min_freq=3, train_ratio=0.8, train=True):
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

        self.bigram_cvec = BigramCounterVector(fname_ngram, min_freq=ngram_min_freq)

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


class LogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


fname = 'data/train/parties_merged.csv'
train_data = 'data/train/train.csv'
test_data = 'data/test/test.csv'
ngram_path = 'ngrams/bigram_merged.csv'

train_dataset = NewsDataset(fname_parties=train_data, fname_ngram=ngram_path, ngram_min_freq=4)
test_dataset = NewsDataset(fname_parties=test_data, fname_ngram=ngram_path, ngram_min_freq=4)

batch_size = 40
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

input_size = train_dataset.get_len_cvec()
print('input_size: ', input_size)

# set hyper parameter
hidden_size = 10000
num_class = 2

learning_rate = 0.0001
num_epoch = 5
# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = LogisticRegression(input_size, hidden_size, num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train model
total_step = len(train_loader)
for epoch in range(num_epoch):
    print('epoch start')
    for i, (data, label) in enumerate(train_loader):
        label = label.squeeze().to(device)
        data = data.to(device)
        outputs = model(data)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 1 == 0:
            print('Epoch : [{}/{}], Step : [{}/{}], Loss : {:.4f}'.format(
                epoch + 1, num_epoch, i + 1, total_step, loss.item()
            ))

    if (epoch + 1) % 1 == 0:
        print('Epoch : [{}/{}], Loss : {:.4f}'.format(
            epoch + 1, num_epoch, loss.item()
        ))

# test model
print('Test model')
total = 0
total_correct = 0
for i, (data, label) in enumerate(test_loader):
    total += len(label)
    label = label.squeeze().to(device)
    data = data.to(device)
    outputs = model(data)
    loss = criterion(outputs, label)

    _, predictions = outputs.max(1)

    is_equal = label == predictions
    num_correct = is_equal.sum()
    total_correct += num_correct

    print('Correct : {}/{}, {}'.format(int(num_correct), len(label), int(num_correct) / len(label)))
    print('Total accuracy {}/{}, {}'.format(int(total_correct), total, int(total_correct) / total))
    print('')

print('---Done---')
