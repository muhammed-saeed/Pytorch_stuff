import pandas as pd
import torch
import csv
import numpy as np
wine_path = '/home/muhammed/Desktop/wine.csv'
winedf = pd.read_csv('/home/muhammed/Desktop/wine.csv')
wine_numpy  = np.loadtxt('/home/muhammed/Desktop/wine.csv',dtype=np.float32, delimiter=";",
skiprows=1)
print('Dataframe wine data labels are ', winedf.columns)
print(len(winedf.columns))
cols = winedf.columns
col_list = cols[:-1]
print('the shape of wine_numpy data is ',wine_numpy.shape)
print(wine_numpy)
wineq = torch.from_numpy(wine_numpy)
data = wine_numpy[:,:-1]
data = np.array(data)
#we get many problems trying convert dataframe into torch tensor
#read csv file using numpy avoids this problem
data = torch.tensor(np.array(data))
print('data shape is ',data.shape)
target = wineq[:,-1].long()
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
#print(target.shape[0])
#print(target_onehot.shape)
#print(target_onehot)

#the code below returns the indexes of bad data
bad_indexes = target <= 3
print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())
bad_data = data[bad_indexes]
bad_data.shape

#to get the bad data itself
bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target > 7]

bad_mean = torch.mean(bad_data, dim = 0)
mid_mean = torch.mean(mid_data, dim = 0)
good_mean = torch.mean(good_data, dim = 0)

col_list = next(csv.reader(open(wine_path), delimiter=';'))

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

print('')

#from the statistic we can get prdiction if the sulphar is 141.3 then wine is good
total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print('Predicted Data')
print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

#print(type(col_list))
#print(col_list)
#print(len(col_list))

#actual data for the wine
actual_indexes = target > 5
print('actual')
#print('length of actual list {:2}, type of actual {:2}, total number of good quality wine{:2}'.format(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()))

print('')
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print('matching')
print('numebr of matches {:2}, matches/prediction {:2}, matches/actual {:2}'.format(n_matches, n_matches / n_predicted, n_matches / n_actual))


