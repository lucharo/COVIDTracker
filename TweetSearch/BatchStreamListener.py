from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from secrets import *
from textblob import TextBlob
import os
import jsonpickle
import dataset
from datafreeze import freeze
import argparse
from tweepy.streaming import StreamListener

print("Setting up twitter authentication...")
# Consumer key authentication
auth = OAuthHandler(consumer_key, consumer_secret)

# Access key authentication
auth.set_access_token(access_token, access_token_secret)

# Set up the API with the authentication handler
api = API(auth, wait_on_rate_limit=True)

print("Authentication done.")

from tweepy.streaming import StreamListener
import json
import time
import sys

class OriginalListener(StreamListener):
    '''This a batch extractor, it extraccts tweets in batches as defined by the batch size parameter'''
    def __init__(self, foldername, batchsize, verbose = False, api = None, fprefix = 'streamer'):
        # set up API
        self.api = api or API()
        self.counter = 0 # number of tweets?
        self.fprefix = fprefix
        self.output  = open('%s/%s_%s.json' % (foldername, self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
        self.batchsize = batchsize
        self.verbose = verbose


    def on_data(self, data):
        if  'in_reply_to_status' in data:
            '''If tweet is is in_reply to another tweet pass the on_status method, which stores the tweet'''
            self.on_status(data)
        elif 'delete' in data:
            '''If tweet was deleted do not return'''
            delete = json.loads(data)['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(json.loads(data)['limit']['track']) is False:
                return False
        elif 'warning' in data:
            warning = json.loads(data)['warnings']
            print("WARNING: %s" % warning['message'])
            return


    def on_status(self, status):
        self.output.write(status)
        if self.verbose:
            tweet = json.loads(status)
            print(tweet["created_at"], tweet["text"])
        self.counter += 1
        if self.counter >= self.batchsize: #tweet batch size
            print("\n\n+"+self.batchsize+" tweets have been streamed and stored\n\n")
            self.output.close()
            self.output  = open('%s/%s_%s.json' % (foldername, self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
            self.counter = 0 # uncomment to keep streaming going
        return


    def on_delete(self, status_id, user_id):
        print("Delete notice")
        return


    def on_limit(self, track):
        print("WARNING: Limitation notice received, tweets missed: %d" % track)
        return


    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return 


    def on_timeout(self):
        print("Timeout, sleeping for 60 seconds...")
        time.sleep(60)
        return 
        
parser = argparse.ArgumentParser(description='Give some words to track on Twitter')
parser.add_argument("--keyword",'-k',type =str, 
                  help='1+ keyword(s) to track on twitter', nargs='+', default = "covid")
parser.add_argument('--output','-o', type=str, action = 'store',
                  help = "Directory name to store streaming results(default: StreamDir)", default = "StreamDir")
parser.add_argument('--batch-size','-bs', type=int, help = 'Size of every bacth of tweets(default = 10000)', default = 10000)
parser.add_argument('--verbose','-v', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
args = parser.parse_args()
print(args)
print("Streaming results with tweets containing: "+', '.join([str(elem) for elem in args.keyword])+
      "; will be stored in the directory "+args.output+"/ . Verbose output is "+"enabled" if args.verbose ==  True else "disabled")

foldername = args.output
if not os.path.exists(foldername):
    os.makedirs(foldername)

keywords_to_track = args.keyword
# Instantiate the SListener object 
listen = OriginalListener(api = api, foldername = args.output, batchsize = args.batch_size, verbose = args.verbose)

# Instantiate the Stream object
stream = Stream(auth, listen)
print("Starting stream, press CTRL-C to stop.\n\n")
# Begin collecting data
stream.filter(track = keywords_to_track) # async allows to use different threads in case
# the current processor runs out of time
