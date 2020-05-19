# inspired from https://www.kaggle.com/clmentbisaillon/classifying-fake-news-with-bert
"""
TODO list:
- use a better optimizer(currently the model just train to answer 0)
- better optimizer(rmsprop?)
- batch/epochs etc : create and use DataLoader
- logits or softmax? the documentation on CrossEntropyLoss is slightly ambiguous
- # TODO:TO LOWER?(maybe tokenizer does this in uits own)
DONE:
- add ana's data to the training
# TODO: schedule learning rate decay
- try 3 classes instead of 2
- try smaller size of the heads network
"""



import os

from collections import OrderedDict # for describing nn to "Sequential" constructor
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertForSequenceClassification
import torch
import torch.nn as torch_nn
import torch.optim as torch_optim
import torch.nn.functional as torch_functional
import torch.utils.data as torch_data

from bert_data_prep import load_and_prepare_data



# == parameters ==


#do_keep_9_label = True
do_keep_9_label = False
nb_classes = 3 if do_keep_9_label else 2

#loss_function_name = 'crossentropy'
loss_function_name = 'mse'

#nb_training_epochs = 2
nb_training_epochs = 10

batch_size = 16


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

model_options =[ nb_classes, ]
# TODO: is that below correct?
"""
if(loss_function_name != 'crossentropy'):
  model_options +=[ 'softmax', ]
  # crossentropy usese the logits so no softmax
"""
model_options +=[ 'softmax' ]

model_options =[ 20, 'relu', nb_classes, 'softmax' ]


if(do_keep_9_label):
  loss_class_weights =[ 1., 10., 1. ]
else :
  loss_class_weights =[ 1., 10. ]

bert_model_name = 'bert-base-uncased'

nit_per_age = 1

# == some checks ==

if(loss_function_name == 'mse'):
  if(nb_classes != 2):
    raise Exception('MSE loss to be used only with 2 classes')


# == data loading and preparation ==

( train_features,
  test_features,
  train_labels,
  test_labels,
  nb_classes ) = load_and_prepare_data(data_file_path,
                                        bert_model_name,
                                        do_convert_label_9_to_0 = not do_keep_9_label,
                                        test_size = 0.18)


print(f'Size of training set: {len(train_features)}')
print(f'Size of testing set: {len(test_features)}')


# == model loading and preparation ==


torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# documentation: https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)

if(loss_function_name == 'mse'):
  bert_model.config.num_labels = 1
else :
  bert_model.config.num_labels = nb_classes
  # no idea what im doing
  bert_model.num_labels = nb_classes


# freeze the pretrained model
for param in bert_model.parameters():
    param.requires_grad = False

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
  elif((layer_option == 'relu') or(layer_option == 'ReLU')):
    layer = torch_nn.ReLU()
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






# == training ==

nb_epochs = nb_training_epochs
learning_rate = 0.01

#torch_data_loader = torch_data.DataLoader(dataset =(train_features, train_labels))
torch_sampler = torch_data.RandomSampler(train_features, replacement = False)
torch_batch_sampler = torch_data.BatchSampler(torch_sampler, batch_size = batch_size, drop_last = True)


def prepare_one_hot_outputs(nb_classes):
  outputs = np.identity(nb_classes)
  return np.array([ torch.tensor(outputs[ i, : ]).float().to(torch_device)
                         for i in range(nb_classes) ])

def prepare_class_index_outputs(nb_classes):
  #return np.array([ torch.tensor([ c, ]).long().to(torch_device) for c in range(nb_classes) ])
  return torch.tensor([[ c, ] for c in range(nb_classes) ]).long().to(torch_device)


if(loss_function_name == 'mse'):
  loss_references = prepare_one_hot_outputs(nb_classes)
  training_outputs = loss_references
else :
  loss_references =  prepare_class_index_outputs(nb_classes)
  training_outputs = loss_references


optimizer = torch_optim.SGD(bert_model.classifier.parameters(), lr = learning_rate)
scheduler = torch_optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5)


loss_accumulator = 0.
loss_history =[]

bert_model.train()

for epoch in range(nb_epochs):
  print(f'Epoch {epoch}/{nb_epochs}')
  batch_count = 0
  for batch_indexes in torch_batch_sampler :
    batch_count += 1
    feature_batch = train_features.values.take(batch_indexes)
    feature_batch = torch.tensor(list(feature_batch)).long().to(torch_device)
    label_batch_values = train_labels.values.take(batch_indexes)
    label_batch = torch.tensor(np.expand_dims(label_batch_values, axis = 1)).long().to(torch_device)
    optimizer.zero_grad()
    # TODO: what is the '[0] output ?
    model_output = bert_model(feature_batch, labels = label_batch)[1]
    if(loss_function_name == 'mse'):
      loss_refs = torch.cat(tuple(loss_references.take(label_batch_values))).reshape(len(batch_indexes), 2)
    else :
      loss_refs = loss_references.take(label_batch)
    loss = loss_function(model_output, loss_refs)
    loss_accumulator += loss.item()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if(batch_count % nit_per_age == 0):
      avg_loss = loss_accumulator / nit_per_age
      print(f'  Age average loss: {avg_loss}')
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
    text = text.to(torch_device)
    label = int(test_labels[data_index])
    model_output = bert_model(text)[0]
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






















