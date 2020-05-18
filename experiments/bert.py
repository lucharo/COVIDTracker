# inspired from https://www.kaggle.com/clmentbisaillon/classifying-fake-news-with-bert
"""
TODO list:
- use a better optimizer(currently the model just train to answer 0)
- better optimizer(rmsprop?)

- logits or softmax? the documentation on CrossEntropyLoss is slightly ambiguous
DONE:
- add ana's data to the training
# TODO: schedule learning rate decay
- try 3 classes instead of 2
- try smaller size of the heads network
"""


from collections import OrderedDict # for describing nn to "Sequential" constructor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split as skl_train_test_split
from sklearn.utils import shuffle as skl_shuffle

from prepare_text import *

# == parameters ==


#do_keep_9_label = True
do_keep_9_label = False

loss_function_name = 'crossentropy'
#loss_function_name = 'mse'

#nb_training_epochs = 2
nb_training_epochs = 10



model_file_name = 'covditracker-bert.pt'

#data_dir_path = '../COVIDTracker/DataToLabel'
data_dir_path = './'
data_file_path = os.path.join(data_dir_path, 'labels.json')


"""
model_options = {
 #'layer_sizes' :[ 20, 'relu', 2, 'softmax' ],
 'layer_sizes' :[ 3 if do_keep_9_label else 2, 'softmax' ],
}
"""

model_options =[ 3 if do_keep_9_label else 2, ]
# TODO: is that below correct?
"""
if(loss_function_name != 'crossentropy'):
  model_options +=[ 'softmax', ]
  # crossentropy usese the logits so no softmax
"""
model_options +=[ 'softmax' ]

if(do_keep_9_label):
  loss_class_weights =[ 1., 10., 1. ]
else :
  loss_class_weights =[ 1., 10. ]


# == data loading and preparation ==

nb_classes = 3 if do_keep_9_label else 2

if(loss_function_name == 'mse'):
  if(nb_classes != 2):
    raise Exception('MSE loss to be used only with 2 classes')

data = pd.read_json(data_file_path, lines = True)
data = data[~ np.isnan(data.label)]

if(not do_keep_9_label):
  data['label'] = data['label'].apply(lambda x : 0. if x == 9. else x)
else :
  data['label'] = data['label'].apply(lambda x : 2. if x == 9. else x)

label_list = np.sort(pd.unique(data['label']))
assert(list(label_list) ==([ 0., 1., 2. ] if do_keep_9_label else[ 0., 1.]))

data = skl_shuffle(data).reset_index(drop = True)

data_features = data['text'].apply(prepare_text)
data_labels = data['label']
del data

train_features, test_features, train_labels, test_labels = skl_train_test_split(data_features, data_labels, test_size = 0.2)
train_features.reset_index(drop = True, inplace = True)
test_features.reset_index(drop = True, inplace = True)
train_labels.reset_index(drop = True, inplace = True)
test_labels.reset_index(drop = True, inplace = True)

print(f'Size of training set: {len(train_features)}')
print(f'Size of testing set: {len(test_features)}')

# == model loading and preparation ==

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as torch_nn
import torch.optim as torch_optim
import torch.nn.functional as torch_functional

torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# documentation: https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
if(loss_function_name == 'mse'):
  bert_model.config.num_labels = 1
else :
  bert_model.config.num_labels = nb_classes
  # no idea what im doing
  bert_model.num_labels = nb_classes

model_input_length = bert_tokenizer.max_model_input_sizes['bert-base-uncased']

# freeze the pretrained model
for param in bert_model.parameters():
    param.requires_grad = False

"""
if(False):
 # Add three new layers at the end of the network
 bert_model.classifier = torch_nn.Sequential(
    torch_nn.Linear(768, 256),
    torch_nn.ReLU(),
    torch_nn.Linear(256, 64),
    torch_nn.ReLU(),
    torch_nn.Linear(64, 2),
    torch_nn.Softmax(dim=1)
 )
"""



if(model_options[-1] not in[ 'softmax', 'Softmax' ]):
  print('WARNING: last layer isnt softmax, this wont probably work!')

layers = len(model_options) *[ None, ]

# TODO: 768, right?
# bert_model.bert.pooler.dense.out_features
prev_layer_size = bert_model.config.hidden_size

for layer_index, layer_option in enumerate(model_options):
  try :
    layer_size = int(layer_option)
  except(ValueError):
    layer_size = -1
  if(layer_size > 0):
    layer = torch_nn.Linear(prev_layer_size, layer_size)
    prev_layer_size = layer_size
  elif((layer_option == 'relu') or(layer_option == 'ReLu')):
    layer = torch_nn.ReLu()
  elif((layer_option == 'softmax') or(layer_option == 'Softmax')):
    layer = torch_nn.Softmax(dim = 1)
  else :
    raise Exception(f'unknown model option: {layer_option}')
  layer_name = f'layer_{layer_index}'
  layers[layer_index] =(layer_name, layer)

bert_model.classifier = torch_nn.Sequential(OrderedDict(layers))

bert_model = bert_model.to(torch_device)

if(loss_function_name == 'mse'):
  loss_function = torch_nn.MSELoss().to(torch_device)
elif(loss_function_name == 'crossentropy'):
  loss_function = torch_nn.CrossEntropyLoss(weight = torch.tensor(loss_class_weights)).to(torch_device)
else :
  raise Exception(f'Unknown loss function {loss_functionn_name}')


# TODO:TO LOWER?(maybe tokenizer does this in uits own)


def prepare_text_for_bert(text):
  return bert_tokenizer.encode(text,
                                  return_tensors = "pt",
                                  max_length = model_input_length).to(torch_device)


nb_epochs = nb_training_epochs
nit_per_period = 10


# == training ==





learning_rate = 0.01

loss_accumulator = 0.
loss_history =[]

bert_model.train()

def prepare_one_hot_outputs(nb_classes):
  outputs = np.identity(nb_classes)
  return[ torch.tensor(outputs[ i, : ]).float().to(torch_device)
           for i in range(nb_classes) ]



if(loss_function_name == 'mse'):
  loss_references = prepare_one_hot_outputs(nb_classes)
  training_outputs = loss_references
else :
  loss_references =  [ torch.tensor([ i, ]).long().to(torch_device)
                       for i in range(nb_classes) ]
  training_outputs =[ torch.tensor([ i, ]).long().to(torch_device)
                       for i in range(nb_classes) ]

"""
if(nb_classes == 2):
  prepare_label = lambda label : torch.tensor([ label, ]).long().to(torch_device)
else :
  prepare_label = lambda label : training_outputs[label]
"""
prepare_label = lambda label : torch.tensor([ label, ]).long().to(torch_device)

learning_rate = 0.01
optimizer = torch_optim.SGD(bert_model.classifier.parameters(), lr = learning_rate)
scheduler = torch_optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5)

for epoch in range(nb_epochs):
  print(f'Epoch {epoch}/{nb_epochs}')
  for data_index, text in enumerate(train_features):
    prepared_text = prepare_text_for_bert(text)
    label = int(train_labels[data_index])
    prepared_label = prepare_label(label)
    optimizer.zero_grad()
    # TODO: what is the '[0] output ?
    model_output = bert_model(prepared_text, labels = prepared_label)[1]
    loss = loss_function(model_output, loss_references[label])
    loss_accumulator += loss.item()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if((data_index+1) % nit_per_period == 0):
      avg_loss = loss_accumulator / nit_per_period
      print(f'  Period average loss: {avg_loss}')
      loss_history.append(avg_loss)
      loss_accumulator = 0.

torch.save(bert_model.state_dict(), model_file_name)

plt.plot(loss_history)
plt.show()

# == evaluation ==

bert_model.eval()

confusion_matrix = np.zeros((nb_classes, nb_classes), dtype = int)

with torch.no_grad():
  for data_index, text in enumerate(test_features):
    prepared_text = prepare_text_for_bert(text)
    label = int(test_labels[data_index])
    prepared_label = torch.tensor([ label, ]).long().to(torch_device)
    model_output = bert_model(prepared_text)[0]
    # note: dont really need to softmax that
    predicted_label = model_output.argmax().item()
    confusion_matrix[label, predicted_label] += 1

nb_test_items = len(test_features)
accuracy = np.trace(confusion_matrix) / nb_test_items

if(nb_classes == 2):
  precision = confusion_matrix[1,1] /(confusion_matrix[0,1] + confusion_matrix[1,1])
  recall = confusion_matrix[1,1] /(confusion_matrix[1,0] + confusion_matrix[1,1])
  #specificity = confusion_matrix[0,0] /(confusion_matrix[0,1] + confusion_matrix[0,0])
else :
  class_sizes = confusion_matrix.sum(1)
  predicted_class_sizes = confusion_matrix.sum(0)
  precisions = np.array([( confusion_matrix[j, j] / predicted_class_sizes[j]
                                if predicted_class_sizes[j] > 0
                                 else 0. )
                               for j in range(nb_classes) ])
  recalls =  np.array([ confusion_matrix[j, j] / class_sizes[j] for j in range(nb_classes) ])
  precision = precisions.mean()
  recall = recalls.mean()

f_score = 2. * precision * recall /(precision + recall)
  

print(f'SCORES:\n  accuracy = {accuracy}\n  precision = {precision}\n  sensitivity = {recall}\n  F-score = {f_score}')






















