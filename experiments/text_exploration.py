
import json
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import os
import pandas as pd
import re
from sklearn.cluster import KMeans as skl_KMeans
from sklearn.decomposition import LatentDirichletAllocation as skl_LatentDirichletAllocation
#from sklearn.decomposition import PCA as skl_PCA
from sklearn.decomposition import TruncatedSVD as skl_TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer as skl_CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as skl_TfidfVectorizer
from sklearn.model_selection import train_test_split as skl_train_test_split
import sqlite3
from textblob import TextBlob

#

keyword_file_path = "keywords-experiments.txt"

#tweet_data_file_dir = "."
tweet_data_file_dir = "/home/alex/projects/pandemic/twita-traeka/COVIDTracker/DataToLabel"
tweet_data_file_dir = "../COVIDTracker/DataToLabel"
#tweet_data_file_dir = os.getcwd()
# NOTE: there should be a separate file for the train and test sets
#tweet_data_file_name = "tweets-experiments.csv"
#tweet_data_file_name = "tweets-experiments.json"
tweet_data_file_name = "alexandre.json"
label_column_name = 'alexandre'

fasttext_file_name = "data-for-fasttext"
fasttext_model_file_path = "fasttext-model.bin"

replacement_text_url = 'url'
replacement_text_mention = 'mn'

tfidf_max_nb_tokens = None
ngram_max = 6

do_check_lang = False
do_process_labelled_data = True

# for starters:
tfidf_max_nb_tokens = 500
ngram_max = 1

nb_clusters = 8

cluster_colors =[ 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange' ]
#

# From https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php
regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# TODO: experimental
regex_mention = r'[@#][0-9a-zA-Z_\-]+'

# rt_filter = " where TweetText not like 'RT %'"


def setup_nltk():
  stopwords = nltk.corpus.stopwords.words('english')
  # when nltk tokenizer is called, it separates do-n't, but n't isnt defined as a stop word
  stopwords.append("n't")
  return set(stopwords), PorterStemmer()

def read_keyword_list(keyword_file_path):
  with open(keyword_file_path) as keyword_file :
    keywords =[ kw.strip() for kw in keyword_file if kw .strip() != '' ]
  return keywords

def replace_urls(document):
  return re.sub(regex_url, replacement_text_url, document)

def replace_mentions(document):
  return re.sub(regex_mention, replacement_text_mention, document)



def prepare_text(document):
  modified_document = document
  modified_document = replace_urls(modified_document)
  modified_document = replace_mentions(modified_document)
  splited_document =[ word.lower() for word in nltk.tokenize.word_tokenize(modified_document) ]
  prepared_content =[ stemmer.stem(t) for t in splited_document if(t not in stopword_set) and(t.isalpha()) ]
  return ' '.join(prepared_content)

  

#

# TOOL SETUP
stopword_set, stemmer = setup_nltk()
keywords = read_keyword_list(keyword_file_path)


# TEXT PREPARATION

#data_tweets = pd.read_csv(os.path.join(tweet_data_file_dir, tweet_data_file_name))[ [ 'text' ] ]
# OLD DATA SOURCE : data_tweets = pd.read_sql('select TweetText, Polarity from TrialOne' + rt_filter, sqlite3.connect(os.path.join(tweet_data_file_dir, tweet_data_file_name)), columns =[ 'TweetText', 'Polarity' ])
#data_tweets = pd.read_json(os.path.join(tweet_data_file_dir, tweet_data_file_name), lines = True)
#tweets_json =[ json.loads(line) for line in open(os.path.join(tweet_data_file_dir, tweet_data_file_name), "r") if line.strip() != "" ]
#data_tweets = pd.DataFrame(tweets_json)

data_file_extension = os.path.splitext(tweet_data_file_name)[-1]

selected_columns =[ 'text' ]

if(do_check_lang):
  selected_columns +=[ 'lang' ]

if(do_process_labelled_data):
  selected_columns +=[ label_column_name ]

if(data_file_extension == '.csv'):
  data_tweets = pd.read_csv(os.path.join(tweet_data_file_dir, tweet_data_file_name), usecols = selected_columns)
elif(data_file_extension == '.json'):
  data_tweets = pd.read_json(os.path.join(tweet_data_file_dir, tweet_data_file_name), lines = True)[ selected_columns ]
else :
  raise Exception("???")


# keeping only english
if(do_check_lang):
  selected_columns.remove('lang')
  data_tweets = data_tweets[data_tweets.lang == 'en'][ selected_columns ]

# removing RT's
data_tweets['RT_filter'] = data_tweets['text'].str.split().str.get(0)
data_tweets = data_tweets[data_tweets.RT_filter != 'RT'][ selected_columns ]


# selected only the labelled data :
if(do_process_labelled_data):
  if(label_column_name != 'label'):
    data_tweets.rename(columns = { label_column_name : 'label', }, inplace = True)
  data_tweets = data_tweets[ data_tweets.label.notna() ]

# prepare the text
#### data_tweetst['text'].str.lower()
data_tweets['prepared_text'] = data_tweets[ ['text'] ].apply(lambda x : prepare_text(x['text']), axis = 1)


# FEATURE CREATION


# new feature : sentiment feature...
# TODO: maybe do this after the filtering of the urls and mentions but before the stemming
data_tweets['polarity'] = data_tweets[ ['text'] ].apply(lambda t : TextBlob(t['text']).sentiment.polarity, axis = 1)

# new feature : keyword occurences
# TODO! do this after filtering the urls and mentions but before stemming
for kw_index, keyword in enumerate(keywords):
  data_tweets[ 'has_keyword_' + str(kw_index) ] = data_tweets[ ['text'] ].apply(lambda t : keyword in t['text'], axis = 1)


# tfidf, ngrams n=1-6
tfidf_vectorizer = skl_TfidfVectorizer(input = 'content',
                                        encoding = 'utf8',
                                        analyzer = 'word',
                                        token_pattern = r'\w{1,}',
                                        ngram_range =(1, ngram_max),
                                        max_features = tfidf_max_nb_tokens)
# TODO:the vectorizer should be fitted only to train data
tfidf_vectorizer.fit(data_tweets['prepared_text'])
tfidf_vectors = tfidf_vectorizer.transform(data_tweets['prepared_text'])














# CLUSTERING/TOPICS ANALYSIS

# clustering:
# TODO: using default parameters, could try other parameterrs
# see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
clusterer = skl_KMeans(n_clusters = nb_clusters)
clusters = clusterer.fit(tfidf_vectors)
cluster_labels = clusters.predict(tfidf_vectors)



PCA_projector = skl_TruncatedSVD(n_components = 3).fit(tfidf_vectors)
projected_data = PCA_projector.transform(tfidf_vectors)


cluster_figure = plt.figure()
plot = cluster_figure.add_subplot(111, projection = '3d')

for cluster_index in range(nb_clusters):
  cluster_color = cluster_colors[cluster_index]
  selected_items = cluster_labels == cluster_index
  cluster_points = projected_data[selected_items]
  plot.scatter(cluster_points[ :, 0 ], cluster_points[ :, 1 ], cluster_points[ :, 2 ], c = cluster_color)

plot.legend(range(nb_clusters))

plt.show()



for cluster_index in range(nb_clusters):
  selected_rows = cluster_labels == cluster_index
  data_tweets[selected_rows][ ['text'] ].to_csv(f'cluster-{cluster_index}.csv')




# LATENT DIRICHLET ALLOCATION

def create_LDA_topics(nb_topics,
                       count_vectors,
                       count_vector_vocab,
                       LDA_learning_method = "online",
                       LDA_max_iter = 1,
                       do_create_summaries = True,
                       nb_top_words = 9):
  LDA_model = skl_LatentDirichletAllocation(
        n_components = nb_topics,
        learning_method = LDA_learning_method,
        max_iter = LDA_max_iter)
  topic_decomposition = LDA_model.fit(count_vectors)
  topic_distribution_over_words = LDA_model.components_
  vocab = np.array(count_vector_vocab)
  if(do_create_summaries):
    topic_summaries = nb_topics *[ [] ]
    for topic_row_i, topic_row in enumerate(topic_distribution_over_words):
      unsorted_top_indexes = np.argpartition(topic_row, - nb_top_words)[ - nb_top_words : ]
      topic_top_indexes = unsorted_top_indexes[
             np.argsort(topic_row[unsorted_top_indexes])][ :: -1 ]
      topic_top_words = vocab[topic_top_indexes]
      topic_summaries[topic_row_i] = " ".join(topic_top_words)
  if(do_create_summaries):
    return topic_decomposition, topic_summaries
  return topic_decomposition


def compute_count_vectors(train_corpus, test_corpus = None):
  count_vector_encoder = skl_CountVectorizer(analyzer = "word", token_pattern = r"\w{1,}")
  count_vector_encoder.fit(train_corpus)
  train_count_vectors = count_vector_encoder.transform(train_corpus)
  if(test_corpus is not None):
    test_count_vectors = count_vector_encoder.transform(test_corpus)
  if(test_corpus is not None):
    return count_vector_encoder, train_count_vectors, test_count_vectors
  return count_vector_encoder, train_count_vectors


count_vector_encoder, count_vectors = compute_count_vectors(data_tweets['prepared_text'])
count_vector_vocabulary = count_vector_encoder.get_feature_names()
topic_decomposition, topic_summaries = create_LDA_topics(8, count_vectors, count_vector_vocabulary, nb_top_words = 21)
for topic_summary in topic_summaries :
  print(topic_summary)


# prepare for fasttext

def save_fasttext_data(file_path, tweet_data):
  fasttext_file = open(file_path, 'w')
  nb_items = len(tweet_data)
  for row_index in range(nb_items):
    row = tweet_data.iloc[row_index]
    label = row['label']
    line = '__label__'
    line += 'no' if(label < 0.5) else('yes' if(label < 1.5) else 'not_sure')
    line += ' '
    fasttext_file.write(line + row['prepared_text'] + '\n')


tweet_train_data, tweet_test_data = skl_train_test_split(data_tweets, test_size = 0.2, shuffle = True)

fasttext_train_file_path = fasttext_file_name + '-train.txt'
save_fasttext_data(fasttext_train_file_path, tweet_train_data)
fasttext_test_file_path = fasttext_file_name + '-test.txt'
save_fasttext_data(fasttext_test_file_path, tweet_test_data)

import fasttext
fasttext_model = fasttext.train_supervised(input = fasttext_train_file_path)
fasttext_model.save_model(fasttext_model_file_path)
# very disappointing:
print(fasttext_model.predict('this is an attempt, do i have the symptoms?'))
print(fasttext_model.predict('i have high fever and dry cough.'))
(nb_samples, precision_at_1, recall_at_1) = fasttext_model.test(fasttext_test_file_path)
print(f'Fasttext test results : precision at 1: {precision_at_1}, recall at 1: {recall_at_1}')

