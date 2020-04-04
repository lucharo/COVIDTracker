from tweepy import OAuthHandler, AppAuthHandler
from tweepy import API, TweepError
from secrets import *
from textblob import TextBlob
import os
import json
import dataset
from datafreeze import freeze
import pandas as pd
import argparse # for command input

print("Authenticating Twitter API keys...")
# Consumer key authentication
auth = AppAuthHandler(consumer_key, consumer_secret) #using appauthhandler to retreive at a faster rate

# Access key authentication
# auth.set_access_token(access_token, access_token_secret)

# Set up the API with the authentication handler
api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify = True)

print("Authentication done.")
import sys
import jsonpickle
import os

parser = argparse.ArgumentParser(description='Give some words to search on Twitter(Retrospective)')
parser.add_argument("--keyword",'-k',type =str, 
                  help='1+ keyword(s) to track on twitter', nargs='*')
parser.add_argument('--output','-o', type=str, action = 'store',
                  help = "Directory name to store streaming results(default: BackSearch)", default = "BackSearch")
parser.add_argument('--file','-f', type=str, action = 'store',
                  help = "File name to store streaming results(default: SomeKeywords), automatically saved to csv", default = "SomeKeywords")
parser.add_argument('--amount','-a', type=int, action = 'store',
                  help = "Number of tweets to search for (default: 10000)", default = 10000)
parser.add_argument('--avoid-retweets','-art', dest='avoidretweets', action='store_false', help="Whether to avoid retweets and only go straight to the source or not (default: True)")
parser.set_defaults(avoidretweets=True)
args = parser.parse_args()

if not args.keyword:
    raise Exception("No keyword entered, please enter keyword to search for")

if not os.path.exists(args.output):
    os.makedirs(args.output)

print(str(args.amount)+ " tweets, containing the term(s): "+', '.join([str(elem) for elem in args.keyword])+"; will be stored in "+args.output+"/"+args.file+".csv") 
  
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

if not args.avoidretweets:
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

else:
    '''
    This script avoids retweets and avoid duplicates as many tweets may be retweeting the same thing
    '''
    print("Avoiding retweets, fetching original tweets only")
    readyFetchedIDs = []
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
                absolutelyNewTweets = 0    
                for tweet in new_tweets:
                    tweetdict = tweet._json # make tweet into a dict
                    while "retweeted_status" in tweetdict: # if the tweet comes from a retweet
                        '''
                        Bare in mind some tweets retweets could be retweets themselves but that is rare
                        #print("RT" if "retweeted_status" in tweetdict["retweeted_status"] else "original")
                        '''
                        originalID = tweetdict["retweeted_status"]["id"]
                        oriTweet = api.get_status(originalID)
                        # overwrite current tweet with original tweet so that original
                        # tweet is stored
                        tweetdict = oriTweet._json 
                    if tweetdict["id"] in readyFetchedIDs:
                        pass # if tweet already stored don't store it
                    else: # if tweet id is new store it and the id to the id  history and add that we have a new tweet
                        f.write(jsonpickle.encode(tweetdict))
                        readyFetchedIDs.append(tweetdict["id"])
                        absolutelyNewTweets +=1
                        
                tweetCount += absolutelyNewTweets
                print("Downloaded {0} tweets".format(tweetCount))
                maxID = new_tweets[-1].id # set maxID to the latest ID in the previous search
            except TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break
    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
    
