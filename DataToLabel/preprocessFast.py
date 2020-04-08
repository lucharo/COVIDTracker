import pandas as pd
import numpy as np
import datetime as dt

# LOAD DF
Symptoms = pd.read_json("Symptoms.json", lines = True)

def preprocess(df):
    # only take those with english as language
    time = [timestamp.strftime("%Y-%m-%d__%H:%M") for timestamp in df['created_at']]
    text = df['full_text']
    try:
        userlocation = [tweet.get('location') for tweet in df['user']]
    except:
        print(tweet)
        
    return pd.DataFrame({"time":np.array(time), "text":text,  "userLocation":np.array(userlocation)})


df = preprocess(Symptoms)

df.to_json("SymptomsClean.json", orient = "records", lines = True)
