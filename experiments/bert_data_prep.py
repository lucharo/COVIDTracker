
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as skl_train_test_split
import torch
from transformers import BertTokenizer

from prepare_text import *


def tokenize_text_series(bert_model_name, text_series):
  bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
  model_input_length = bert_tokenizer.max_model_input_sizes['bert-base-uncased']
  return text_series.apply(lambda t : bert_tokenizer.encode(t, return_tensors = 'pt', max_length = model_input_length))


def load_and_prepare_data(data_file_path,
                           bert_model_name,
                           do_convert_label_9_to_0 = True,
                           test_size = 0.18):
  nb_classes = 2 if do_convert_label_9_to_0 else 3
  data = pd.read_json(data_file_path, lines = True)
  data = data[~ np.isnan(data.label)]
  data['label'] = data['label'].apply(np.long)
  if(not do_keep_9_label):
    data['label'] = data['label'].apply(lambda x : 0 if x == 9 else x)
  else :
    data['label'] = data['label'].apply(lambda x : 2 if x == 9 else x)
  label_list = np.sort(pd.unique(data['label']))
  assert(list(label_list) ==([ 0, 1 ] if do_convert_label_9_to_0 else[ 0, 1, 2 ]))
  data_features = data['text'].apply(prepare_text)
  data_labels = data['label']
  del data
 ( train_features_df,
    test_features_df,
    train_labels_df,
    test_labels_df ) = skl_train_test_split(data_features, data_labels, test_size = test_size)
  train_features = tokenize_text_series(train_features_df['text'])
  test_features = tokenizer_text_series(test_features_df['text'])
  train_labels = train_labels_df.apply(lambda r : torch.tensor(row['label']).long())
  test_labels = test_labels_df.apply(lambda r : torch.tensor(row['label']).long())
  del train_features_df
  del test_features_df
  del train_labels_df
  del test_labels_df
  return train_features, test_features, train_labels, test_labels, nb_classes










