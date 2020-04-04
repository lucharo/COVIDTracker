from tweepy import OAuthHandler, AppAuthHandler
from tweepy import API, TweepError
from secrets import *
from textblob import TextBlob
import os
import json
import dataset
from datafreeze import freeze
import pandas as pd

# Consumer key authentication
auth = AppAuthHandler(consumer_key, consumer_secret) #using appauthhandler to retreive at a faster rate

# Access key authentication
# auth.set_access_token(access_token, access_token_secret)

# Set up the API with the authentication handler
api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify = True)

import sys
import jsonpickle
import os

parser = argparse.ArgumentParser(description='Give some words to search on Twitter(Retrospective)')
parser.add_argument("--keyword",'-k',type =str, 
                  help='1+ keyword(s) to track on twitter', nargs='+', default = "covid")
parser.add_argument('--output','-o', type=str, action = 'store',
                  help = "Directory name to store streaming results(default: BackSearch)", default = "BackSearch")
parser.add_argument('--file','-f', type=str, action = 'store',
                  help = "File name to store streaming results(default: SomeKeywords), automatically saved to csv", default = "SomeKeywords")
parser.add_argument('--amount','-a', type=int, action = 'store',
                  help = "Number of tweets to search for (default: 10000)", default = 10000)
args = parser.parse_args()

searchQuery = args.keyword
maxTweets = args.amount
tweetsPerQry = 100 # max set by twitter api
fName = args.output+"/"+args.file+".csv"

# the api search output SearchObject is an object that can be turned into a list of json str with the _json method
# the search object itself can be indexed like a list
sinceID = None # no lower limit (as far as necessary)
maxID = -1 #exploit negative indexing, last tweet
tweetCount = 0
# the ID is just an ordered number, If I tweet now and my tweet ID is 1 and
# your tweet comes next your tweet id will be 2


print("Downloading max {0} tweets".format(maxTweets))

with open(fName, 'w') as f:
    while tweetCount < maxTweets:
        try:
            if (maxID<=0):
                # if your maxID is negative (1st iteration) do this
                if (not sinceID):
                    new_tweets = api.search(q=searchQuery, count = tweetsPerQry)
                else: # if sinceID exists set sinceID as that minimum
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceID)
            else: # if maxID is positive ? 
                if (not sinceID): # if sinceID not defined don't set low boundary
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(maxID - 1))
                else: # if sinceID exist set low and up boundary
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(maxID - 1),
                                            since_id=sinceID)
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                f.write(jsonpickle.encode(tweet._json))
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            maxID = new_tweets[-1].id # set maxID to the latest ID in the previous search
        except TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break
print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))