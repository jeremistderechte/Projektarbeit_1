import pandas as pd
import en_core_web_sm
import re


def cleanTwitter(sentences):
    for i, comment in enumerate(comments):
        comment = re.sub("@", "", comment)
        comments[i] = comment
    return comments

nlp = en_core_web_sm.load()

print(nlp.get_pipe('ner').labels)

df = pd.read_csv("./datasets/twcs.csv")

comments = list(pd.read_csv("./datasets/twcs.csv").sample(500)["text"])#["commentBody"]
comments = cleanTwitter(comments)

newCommentsNLP = []
for sentence in comments:
    newCommentsNLP.append(nlp(sentence))
    
df_entity = pd.DataFrame(columns=['index','word','range','label_name'])

wordList = []
startRangeList = []
endRangeList = []
labelNameList = []
indexList = []

for  i, sentence in enumerate(newCommentsNLP):
    for entity in sentence.ents:
        if entity.label_ == "PERSON":
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
    