import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import NewsDataset
from models import LogisticRegression

fname = 'data/train/parties_merged.csv'
train_data = 'data/train/train.csv'
test_data = 'data/test/test.csv'
ngram_path = 'ngrams/bigram_merged.csv'

train_dataset = NewsDataset(
    fname_parties=train_data, fname_ngram=ngram_path, ngram_min_freq=4)
test_dataset = NewsDataset(
    fname_parties=test_data, fname_ngram=ngram_path, ngram_min_freq=4)

batch_size = 40
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
                epoch + 1, num_epoch, i + 1, total_step, loss.item()))

    if (epoch + 1) % 1 == 0:
        print('Epoch : [{}/{}], Loss : {:.4f}'.format(epoch + 1, num_epoch,
                                                      loss.item()))

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

    print('Correct : {}/{}, {}'.format(
        int(num_correct), len(label),
        int(num_correct) / len(label)))
    print('Total accuracy {}/{}, {}'.format(
        int(total_correct), total,
        int(total_correct) / total))
    print('')

print('---Done---')
