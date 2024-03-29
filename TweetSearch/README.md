# HOW TO USE COVID TRACKER

## Setting up repo
<details>
 <summary>Expand</summary>
 
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
</details>

## Using `BatchStreamListener.py`
<details>
 <summary>Expand</summary>
 
This script can be used to start a stream with a given set of keywords and a directory output. Each 10,000 tweets will be stored in separate json files within the output directory. If no output directory is given the results will be stored in `StreamDir/`. For help with script commands type: `BatchStreamListener.py -h` in your command line.
Options are:
  * `--keyword, -k`: (no default) keywords to search for can be a single word or a list.
  * `--output, -o`: (default: StreamDir/) Output directory to store results
  * `--verbose, -v`: (default: False) whether output should be verbose or not (if verbose, tweet creation time and text will be displayed)
  * `--batch-size, -bs`: (default: 10000) Number of tweets to store in each individual file 
  * `--location, -l`: (no default) [type: string] Name of a region that has been stored in RegionCoords.json. To learn how to add a region to RegionCoords.json read further down.
  * `--coordinates, -cord`: (no default)[type: float][format: [SWlongitude SWlatitute NElongitude NElatitude]x (as many region as you want to track)] Custom region defined by polygon coordinates to track. You should give coordinates of the southwest corner first and then the coordinates or the northeast corner as this is the required format by the Twitter API.

Example coordinates are:
```
    LOCATIONS = [-124.7771694, 24.520833, -66.947028, 49.384472,        # Contiguous US
                 -164.639405, 58.806859, -144.152365, 71.76871,         # Alaska
                 -160.161542, 18.776344, -154.641396, 22.878623]        # Hawaii
```

* Run Tweet Batch sampler
```
python BatchStreamListener.py --keyword key1 key2 ... keyN --output OutputDirName --batch-size 10000 --verbose --location london 
```
Stop the stream anytime by clicking CTRL-C. Now your data has been loaded in the output directory. To load the json files in python:
### Loading results in python/jupyter notebook
Just input the right output directory
```
import os
import json
import pandas as pd

tweets = []
filedir = "OUTPUT_DIRECTORY"
for file in os.listdir(filedir):    
    for tweet in open(os.path.join(filedir,file), 'r'):
        tweets.append(json.loads(tweet))

tweets = pd.DataFrame(tweets)
```
</details>

## Using `SearchBack.py`
<details>
 <summary>Expand</summary>
 
This script can be used to search for tweets (limited to 1% of all Twitter) that contain a certain keyword or a certain location. Similarly to `BatchStreamListener.py`, this script can be called from the command line with a few options.
By default, the script will look for 10,000 tweets that contain the teerm "covid" (case insensitive) and store the results in a csv file in the directory BackSearch/ in the file SomeKeywords.json
Options are:
  * `--query, -q`: (no default) query to start search. Please read [Twitter standard search operators](https://developer.twitter.com/en/docs/tweets/rules-and-filtering/overview/standard-operators) before starting a search. An understanding of [the order of precedence of logical operators](https://www.bouraspage.com/repository/algorithmic-thinking/what-is-the-order-of-precedence-of-logical-operators) help.
  * `--output, -o`: (default: BackSearch) Output directory to store results
  * `--file, -f`: (default: SomeKeywords), File name of json file where all search results will be stored
  * `--output-format, -of`: (default: json)[other options csv] if csv option is called, the search output will be stored in a csv file in addition to the json file.
  * `--amount, -a`: (default: 10000), Number of tweets to search 
  * `--include-retweets, -rt`: (default: False) whether to include retweets in search results or not. 

* Run `SearchBack.py`
```
python SearchBack.py --query "(key1 AND key2) OR key3 -key4)" --output OutputDirName --file OutputFileName --amount 10000 --include-retweets 
```

Many queries are possible as well as filtering of retweets or selecting only tweets in certain languages. __Some example searches:__
```
# tweets containining the terms trump OR sanders AND covid
-q "(trump OR sanders) AND covid" -a 1000 
# tweets containing either of these symptoms, 'loss of smell' will match exactly loss of smell and lang:en will ensure we only get tweets in english
-q "cough OR cold OR symptom OR symptoms OR fever OR 'loss of smell' lang:en"

```

### To load the output file from `SearchBack.py`

```
import pandas as pd
import json

tweets = pd.read_json("path/to/file.json", lines = True)
```
</details>

## Storing new custom region coordinates
<details>
 <summary>Expand</summary>
 
Twitter has a strict format for location search. To make it easier to track any custom region I have made a script (`DumpingGeoJSONs.py`). The script should be run from the command window. The script will prompt you to enter a region name and thereafter GeoJSON file (raw). The workflow is as follows, go to [http://geojson.io/](http://geojson.io/), select a custom a region in the world map. Then copy the raw json text from the right panel and paste it in the command prompt once you are requested to do so.
Now your new custom region will be stored in the file `RegionCoords.json`, if the file does not exist it will be created.
</details>
