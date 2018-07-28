from tools import csv2dictlist, dictlist2csv
import random

# open data file and convert to list of dict
all_dict = csv2dictlist('data/train/parties_merged.csv')

# shuffle data randomly
random.shuffle(all_dict)

# calculate number of train and test
total = len(all_dict)
num_train = int(total * 0.8)
num_test = total - num_train
print('total:', total)
print('num_train:', num_train)
print('num_test:', num_test)

# split data into two
train = all_dict[:num_train]
test = all_dict[num_train:]

# save as csv file
dictlist2csv(dict_list=train, out_name='train.csv')
dictlist2csv(dict_list=test, out_name='test.csv')
