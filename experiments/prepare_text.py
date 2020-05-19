
import nltk
import re



regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
regex_mention = r'[@#][0-9a-zA-Z_\-]+'

replacement_text_url = 'url'
replacement_text_mention = 'mn'

def replace_urls(document):
  return re.sub(regex_url, replacement_text_url, document)

def replace_mentions(document):
  return re.sub(regex_mention, replacement_text_mention, document)


def prepare_text(document):
  modified_document = document
  modified_document = replace_urls(modified_document)
  modified_document = replace_mentions(modified_document)
  splited_document =[ word.lower() for word in nltk.tokenize.word_tokenize(modified_document) ]
  return ' '.join(splited_document)
