#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:38:00 2023

@author: jeremy
"""
import re
from flask import Flask, render_template, request
import pandas as pd
import pickle

#data = pd.read_csv("./datasets/twitterDataCleaned.csv")["text"]

data = pd.read_csv("./datasets/new_york_times_cleanend.csv")["commentBody"]

app = Flask(__name__)

current_row_index = 0
labelIndex = 0
indexList = []
labelList = []
    

@app.route("/", methods=["GET", "POST"])
def display_row():
    global current_row_index
    global indexList
    global labelList
    global labelIndex
    if request.method == 'POST':
        label = request.form["label"]
        if label != "":
            labelIndex += 1
        indexList.append(current_row_index)
        labelList.append(label)
        
        with open("index", "wb") as f:
            pickle.dump(indexList, f)
            
        with open("label", "wb") as f:
            pickle.dump(labelList, f)

        current_row_index += 1
        
    return render_template('nerLabelTool.html', sentence=data[current_row_index], index=(current_row_index+1), indexLabel=labelIndex)
if __name__ == '__main__':
    app.run()




        







# =============================================================================
# count = 0
# for match in re.finditer(r'das', string):
#     count += 1
#     print("match", count, match.group(), "start index", match.start(), "End index", match.end())
# =============================================================================