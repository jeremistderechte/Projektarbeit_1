import spacy
#import spacy_transformers
#from spacy.cli import download
import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.training import Example
import json
import pickle

class custom_model:
    def __init__(self):
        self.data = None
        self.model = None


    def __create_pipeline(self, model_architecture):
        model = None
        if model is not None:
            if (model_architecture == "standard"):
                nlp = spacy.load(model)
            else:
                nlp = spacy.load(model, enable=["transformers"])
            print("Loads model '%s'" % model)
        else:
            nlp = spacy.blank("en")
            print("Created blank 'en' model")
        if 'ner' not in nlp.pipe_names:
            if (model_architecture == "transformer"):
                nlp.add_pipe("transformer")
            ner = nlp.add_pipe("ner")
        else:
            print("get pipeline")
            ner = nlp.get_pipe("ner")
        return ner, nlp

    def load(self, path_to_data):
        print("Loading data... Do or do not. There is no try")
        self.data = pd.read_csv(path_to_data)  # "./datasets/New_York_Times_labeled_dataframe_manual_cleaned.csv"
        self.data = self.data[self.data["ranges"].isna() == False]

    def train(self, model_architecture, iterations):

        print("Training started... I feel a great disturbance in the force")

        training_test_text = []

        training_test_entity = []

        for index, row in self.data.iterrows():
            text = row["sentence"]
            range_word = json.loads(row["ranges"])
            temp_list = []

            if len(range_word) == 1:
                range_word = range_word[0]
                range_word.append("PER_CUSTOM")
                entity_dict = {"entities": [tuple(range_word)]}
            else:
                for i, inner_range in enumerate(range_word):
                    inner_range.append("PER_CUSTOM")
                    temp_list.append(tuple(inner_range))
                range_word = temp_list
                entity_dict = {"entities": tuple(range_word)}
            training_test_text.append(text)
            training_test_entity.append(entity_dict)

        train_text, test_text, train_entity, test_entity = train_test_split(training_test_text, training_test_entity,
                                                                            test_size=0.2, shuffle=False)

        ner, nlp = self.__create_pipeline(model_architecture)
        n_iter = iterations
        loss_graph_list = []
        #test = nlp.pipe_names
        if (model_architecture == "standard"):
            other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
            with nlp.disable_pipes(other_pipes):
                optimizer = nlp.begin_training()
        else:
            optimizer = nlp.begin_training()
        for itn in range(n_iter):
            losses = {}
            for text, annotations in zip(train_text, train_entity):
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], losses=losses)

            print("Iteration", itn + 1, "Losses", losses)
            loss_graph_list.append(losses)
        self.model = nlp

    def save_model(self, save_path):
        print("Saving model to " + save_path)
        print("You can't stop change any more than you can stop the suns from setting.")
        with open(save_path, "wb") as f:
           pickle.dump(self.model, f)
