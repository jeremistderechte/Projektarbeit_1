import en_core_web_sm
#import en_core_web_trf
import pickle
import pandas as pd
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training.example import Example
import spacy
import ast
import re

class model_to_evaluate:
    def __init__(self):
        if spacy.prefer_gpu():
            print("Using GPU")
        else:
            print("Using CPU")
        self.data = None
        self.nlp = None
        self.labelname = ""

    def load_data(self, path_to_data):
        self.data = pd.read_csv(path_to_data)

    def load_model(self, path_to_model, is_selftrained): # Spacy (PERSON), custom (PER_CUSTOM)
        # Load trained model
        if is_selftrained:
            with open(path_to_model, "rb") as f:
                self.nlp = pickle.load(f)
            self.labelname = "PER_CUSTOM"
        else:
            self.nlp = spacy.load("en_core_web_sm")
            self.labelname = "PERSON"

    def evaluate(self):

        #comments = self.data["sentence"]

        newCommentsNLP = []
        #lengthDataSet = len(comments)
        #percentage = -1

        scorer = Scorer()
        example = []
        for sentence, entity in zip(self.data["sentence"], self.data["label"]):
            if self.labelname == "PERSON":
                entity = re.sub("PER_CUSTOM", "PERSON", entity)
            pred = self.nlp(sentence)
            newCommentsNLP.append(pred)
            doc = self.nlp.make_doc(sentence)
            temp = Example.from_dict(pred, ast.literal_eval(entity))
            example.append(temp)
        scores = scorer.score(example)
        print(scores["ents_per_type"])


        # for i, sentence in enumerate(comments):
        #    if not isinstance(sentence, str):
        #        sentence = ""
        #    newCommentsNLP.append(nlp(sentence))
        #    percentageNew = round(i / lengthDataSet,2) * 100
        #    if percentageNew != percentage:
        #        percentage = percentageNew
        #        print(round(percentage,1), "%")

        df_entity = pd.DataFrame(columns=['index', 'word', 'range', 'label_name'])

        wordList = []
        startRangeList = []
        endRangeList = []
        labelNameList = []
        indexList = []

        for i, sentence in enumerate(newCommentsNLP):
            for entity in sentence.ents:
                if entity.label_ == self.labelname:  # Spacy (PERSON), custom (PER_CUSTOM)
                    wordList.append(entity.text)
                    startRangeList.append(entity.start_char)
                    endRangeList.append(entity.end_char)
                    labelNameList.append(entity.label_)
                    indexList.append(i)

        rangeList = []

        for start, end in zip(startRangeList, endRangeList):
            tempList = []
            tempList.append(start)
            tempList.append(end)
            rangeList.append(tempList)

        df_entity["index"] = indexList
        df_entity["word"] = wordList
        df_entity["range"] = rangeList
        df_entity["label_name"] = labelNameList
