
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer as skl_TfidfVectorizer
import sqlite3
from textblob import TextBlob

"""
 tweet : "location" + text
"""

keyword_file_path = "keywords.txt"

tweet_data_file_dir = "."
# NOTE: there should be a separate file for the train and test sets
#tweet_data_file_name = "tweets-test.csv"
tweet_data_file_name = "tweets.db"


replacement_text_url = 'url'
replacement_text_mention = 'mn'

tfidf_max_nb_tokens = None
ngram_max = 6


# for starters:
tfidf_max_nb_tokens = 500
ngram_max = 1


#

# From https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php
regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# TODO: experimental
regex_mention = r'[@#][0-9a-zA-Z_\-]+'

rt_filter = " where TweetText not like 'RT %'"


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

# necessary
stopword_set, stemmer = setup_nltk()
keywords = read_keyword_list(keyword_file_path)


#data_tweets = pd.read_csv(os.path.join(tweet_data_file_dir, tweet_data_file_name))[ [ 'text' ] ]
data_tweets = pd.read_sql('select TweetText, Polarity from TrialOne' + rt_filter, sqlite3.connect(os.path.join(tweet_data_file_dir, tweet_data_file_name)), columns =[ 'TweetText', 'Polarity' ])
data_tweets.rename(columns = { 'TweetText' : 'text', 'Polarity' : 'polarity' }, inplace = True)


# prepare the text
data_tweets['prepared_text'] = data_tweets[ ['text'] ].apply(lambda x : prepare_text(x['text']), axis = 1)



# new feature : sentiment feature...
# TODO: maybe do this after the filtering of the urls and mentions but before the stemming
# TODO: this is already provided
#data_tweets['polarity'] = data_tweets[ ['text'] ].apply(lambda t : TextBlob(t['text']).sentiment.polarity, axis = 1)

# new feature : keyword occurences
"a set of flu-related keywords/terms were used as a set of features for flu-related tweets. The list includes some important influenza-related keywords, symptoms, and treatments"
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
tfidf_vectorizer.fit(data_tweets['prepared_text'])
tfidf_vectors = tfidf_vectorizer.transform(data_tweets['prepared_text'])



