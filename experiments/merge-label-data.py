import numpy as np
import os
import pandas as pd

data_dir_path = '../COVIDTracker/DataToLabel'
data_file_path = os.path.join(data_dir_path, 'alexandre.json')
data_alexandre = pd.read_json(data_file_path, lines = True)
data_file_path = os.path.join(data_dir_path, 'Ana_labels.json')
data_ana = pd.read_json(data_file_path, lines = True)

#data.rename(columns = { 'alexandre' : 'label' }, inplace = True)
#data_ana.rename(columns = { 'Ana' : 'label' }, inplace = True)




data = pd.merge(data_alexandre, data_ana, on =['time', 'text', 'userLocation', 'Luis'])

alexandre = ~ np.isnan(data['alexandre'])
ana_labels = np.where(alexandre, 0., data['Ana'])
ana = ~ np.isnan(data['Ana'])
alexandre_labels = np.where(ana, 0., data['alexandre'])

label_indexes = np.array(alexandre, dtype = int)
labels = label_indexes * alexandre_labels +(1. - label_indexes) * ana_labels

np.sum(~np.isnan(labels))

data['label'] = labels
data.drop(columns=['Luis','alexandre','Ana'],inplace=True)

data.to_json('labels.json', orient = "records", lines = True)

