#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np
import datetime as dt
import random 
import os
import sys


# In[86]:


symptoms = pd.read_json("SymptomsClean.json", lines = True)


# In[87]:


symptoms


# In[88]:


labeller = input("Hello kind labeller, what's your name? ")


# In[89]:


if labeller not in symptoms.columns:
    print("\nThanks for joining "+labeller+", a new column with your name has been created.")
    symptoms[labeller] = np.nan
else:
    # Figures for given labeller
    # indices for columns that have not yet been labelled
    yet_to_label = symptoms[symptoms[labeller].isna()].index.to_list()
    percentage_progress = 100*(len(symptoms) - len(yet_to_label))/len(symptoms)
    
    print("\nWelcome back "+labeller+", resume labelling where you left it!\n"+
         "Luis's Progress: "+str((len(symptoms) - len(yet_to_label)))+"/"+
          str(len(symptoms))+" ("+str(round(percentage_progress,2))+"%)")


# In[90]:


input("\n\n INSTRUCTIONS:\n\tA total of "+
      str(symptoms.shape[0])+
      " tweets with words containing COVID19 symptoms such as cough, fever or loss of smell have been collected."+
     " You will be presented the text from tweets you have not yet labelled and will be prompted to report whether "+
     "tweet consists of someone reporting symptoms (label = 1), no symptoms (e.g. news reporting newly discovered "+
     "symptoms; label = 0) or the text is ambiguous (label = 9) or delete the entry if you find it irrelevant (PRESS 5)\n\nONCE YOU ARE READY TO START, PRESS ENTER.\n\nFINISH AT ANY TIME BY ENTERING A NEGATIVE NUMBER\n")


# In[94]:


class LabelPrompter():
    
    def __init__(self, labeller):
        self.labeller = labeller
        self.yet_to_label = symptoms[symptoms[labeller].isna()].index.to_list()
    
    def prompt(self):
        random.shuffle(self.yet_to_label)
        for tweet_index in self.yet_to_label:
            while np.isnan(symptoms.loc[tweet_index, self.labeller]):
                print("Index: "+str(tweet_index))
                print(symptoms.loc[tweet_index, "text"])
                label = int(input("Report label:\n Symptom - 1\n No Symptom 0\n"+
                                  " Ambiguous - 9\n DELETE - 5\n LEAVE NOW - < 0\n"))
                if label in [0,1,9]:
                    # update value
                    symptoms.loc[tweet_index, self.labeller] = label
                    # update list
                    self.yet_to_label = symptoms[symptoms[self.labeller].isna()].index.to_list()
                    random.shuffle(self.yet_to_label)
                    # save results
                    symptoms.to_json("SymptomsClean.json", orient = "records", lines = True)
                    os.system("clear")
                elif label == 5:
                    print("Tweet deleted")
                    symptoms.drop([tweet_index], inplace = True)
                    #remove from yet_to_label too
                    yet_to_label.remove(tweet_index)
                    #update stuff
                    self.yet_to_label = symptoms[symptoms[self.labeller].isna()].index.to_list()
                    random.shuffle(self.yet_to_label)
                    # save results
                    symptoms.to_json("SymptomsClean.json", orient = "records", lines = True)
                    os.system("clear")
                    break #break the while loop
                elif label <0:
                    print("Thanks for your work today!")
                    sys.exit()
                else:
                    print("Please enter a valid number(0, 1 or 9)!")


# In[95]:


prompter = LabelPrompter(labeller)


# In[96]:


prompter.prompt()


# In[ ]:




