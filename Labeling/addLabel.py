#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:27:36 2023

@author: jeremy
"""
import pickle
import pandas as pd

data = pd.read_csv("./twitterDataCleaned.csv")["text"].head(1000)

    
with open('label_new', "rb") as f:
    labelList = pickle.load(f)
     
with open("label_new", "wb") as f:
     pickle.dump(labelList, f)
