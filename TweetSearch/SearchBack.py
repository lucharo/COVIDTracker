from tweepy import OAuthHandler, AppAuthHandler
from tweepy import API, TweepError
from secrets import *
from textblob import TextBlob
import os
import json
import dataset
from datafreeze import freeze
import pandas as pd
import csv
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

parser.add_argument("--query",'-q',type =str, 
                  help='Query to use for search(the use of brackets is important), for more info visit: https://developer.twitter.com/en/docs/tweets/rules-and-filtering/overview/standard-operators', nargs='*')

parser.add_argument('--output','-o', type=str, action = 'store',
                  help = "Directory name to store streaming results(default: BackSearch)", default = "BackSearch")

parser.add_argument('--file','-f', type=str, action = 'store',
                  help = "File name to store streaming results(default: SomeKeywords), automatically saved to csv", default = "SomeKeywords")

parser.add_argument('--output-format', '-of', help = "Output format of choice (default: json)", type = str, default = "json")

parser.add_argument('--amount','-a', type=int, action = 'store', help = "Number of tweets to search for (default: 10000)", default = 10000)

parser.add_argument('--include-retweets','-rt', dest='include_retweets', action='store_true', help="Whether to include retweets or not (default: False)")

parser.set_defaults(include_retweets=False)
args = parser.parse_args()

if not args.query:
    raise Exception("No query entered, please enter query to start search")
else:
    args.query = ''.join(args.query)

if not os.path.exists(args.output):
    os.makedirs(args.output)

print(args.query)
print(type(args.query))
retweet_notice = " not " if not args.include_retweets else " "
print(str(args.amount)+ " tweets, using the query: "+str(args.query)+
    "; will be stored in "+args.output+"/"+args.file+".json and retweets will"+ retweet_notice + "be fetched in your search.") 
  
searchQuery = args.query
maxTweets = args.amount
tweetsPerQry = 100 # max set by twitter api
fName = args.output+"/"+args.file+".json"

if not args.include_retweets:
    searchQuery += " -filter:retweets"
print("Your query is: " + searchQuery)
# the api search output SearchObject is an object that can be turned into a list of json str with the _json method
# the search object itself can be indexed like a list
sinceID = None # no lower limit (as far as necessary)
maxID = -1 #exploit negative indexing, last tweet
tweetCount = 0
# the ID is just an ordered number, If I tweet now and my tweet ID is 1 and
# your tweet comes next your tweet id will be 2


print("Downloading max {0} tweets".format(maxTweets))
firstTweet = True
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
                # a json file is basically one json str after the other
                # a json string itselff it's just a dictionary in text format
                f.write(json.dumps(tweet._json)+"\n") # tweet._json gets you the tweet as a dict
                # json.dumps gets you the tweet from a dict to a json str
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            maxID = new_tweets[-1].id # set maxID to the latest ID in the previous search
        except TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break            
print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

if args.output_format == "csv":
    tweets = []
    for line in open(args.output+"/"+args.file+".json", "r"):
        tweets.append(json.loads(line))
    tweets = pd.DataFrame(tweets)
    tweets = tweets.iloc[:,1:]
    tweets.to_csv(args.output+"/"+args.file+".csv")

    print("For convenience, a csv copy of the file was saved.")

