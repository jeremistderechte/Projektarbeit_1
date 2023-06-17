#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:33:58 2023

@author: jeremy
"""
import pickle
import pandas as pd
import string
import re

data = pd.read_csv("./datasets/twitterDataCleaned.csv")["text"]

with open('index', "rb") as f:
    indexList = pickle.load(f)
    
with open('label', "rb") as f:
    labelList = pickle.load(f)
    
dataframeLabelTwitter = pd.DataFrame(columns=["sentence", "labels"])    
    

sentenceList = []
for index in indexList:
    sentenceList.append(data.iloc[index])

dataframeLabelTwitter["sentence"] = sentenceList
dataframeLabelTwitter["labels"] = labelList       
        
    
    
    
# =============================================================================
#     
# def cleanData(data):
#     for i, sentence in enumerate(data):
#         sentence = ' '.join(re.sub("([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)"," ", sentence).split())
#         #sentence = sentence.translate(str.maketrans("", "", string.punctuation)) 
#         sentence = re.sub(r"\s+", " ", sentence)
#         sentence = sentence.lower()
#         data[i] = sentence
#     return data
#         
# 
#     
# twitterData = pd.read_csv("./datasets/twcs.csv")["text"].sample(frac=1).reset_index(drop=True)
# 
# twitterData = cleanData(twitterData)
# 
# twitterData.to_csv("twitterDataCleaned.csv")
# =============================================================================

