# HOW TO USE COVID TRACKER

## Setting up repo
1. Clone repo
```
git clone https://github.com/lc5415/COVIDTracker.git
```
2. create virtualenv with requirements
```
cd TweetSearch
python3 -m venv tweetCOVID
source tweetCOVID/bin/activate
pip install -r requirements.txt
```
## Using `BatchStreamListener.py`
This script can be used to start a stream with a given set of keywords and a directory output. Each 10,000 tweets will be stored in separate json files within the output directory. If no output directory is given the results will be stored in `StreamDir/`. For help with script commands type: `BatchStreamListener.py -h` in your command line.
Options are:
  * `--keyword, -k`: (default: covid) keywords to search for can be a single word or a list.
  * `--output, -o`: (default: StreamDir/) Output directory to store results
  * `--verbose, -v`: (default: False) whether output should be verbose or not (if verbose, tweet creation time and text will be displayed)
  * `--batch-size, -bs`: (default: 10000) Number of tweets to store in each individual file 
  
* Run Tweet Batch sampler
```
python BatchStreamListener.py --keyword key1 key2 ... keyN --output OutputDirName --batch-size 10000 --verbose
```

## Using `SearchBack.py`
This script can be used to search for tweets (limited to 1% of all Twitter) that contain a certain keyword or a certain location. Similarly to `BatchStreamListener.py`, this script can be called from the command line with a few options.
By default, the script will look for 10,000 tweets that contain the teerm "covid" (case insensitive) and store the results in a csv file in the directory BackSearch/ in the file SomeKeywords.csv 

* Run `SearchBack.py`
```
python SearchBack.py --keyword key1 key2 ... keyN --output OutputDirName --file OutputFileName --amount 10000 
```

