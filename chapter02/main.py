### UNDERSTANDING NATURAL LANGUAGE ###

#%% WORD VECTOR
import spacy

# python -m spacy download en_core_web_lg
# nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('pt_core_news_lg')

doc = nlp('Oi, você pode me ajudar?')

# Token and its vectors represantation build with GloVe algorithm
for token in doc:
    print("{}: {}".format(token, token.vector[:3]))

# Similarities between documents
doc = nlp("gato")

doc.similarity(nlp("posso"))
doc.similarity(nlp("cachorro"))


#%% INTENTS AND CLASSIFICATION
import numpy as np
import pandas as pd
import spacy
import os

this_file = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(this_file))
data_dir = os.path.join(base_dir, 'datasets')

# Load dataset
atis_train = pd.read_csv(os.path.join(data_dir, 'atis_intents_train.csv'))
labels_train = atis_train['intents'].tolist()
sentences_train = atis_train['sentences'].tolist()

# Load spacy model
nlp = spacy.load('en_core_web_lg')

X_train_shape = (len(sentences_train), nlp.vocab.vectors_length)
X_train = np.zeros(X_train_shape)

for idx, sentence in enumerate(sentences_train):
    X_train[idx, :] = nlp(sentence).vector


# Nearest neighbor classification
from sklearn.metrics.pairwise import cosine_similarity

test_message = """
i would like to find a flight from charlotte
to las vegas that makes a stop in st. louis
"""

test_x = nlp(test_message).vector

scores = [
    cosine_similarity(X_train[i, :].reshape(-1, 300), test_x.reshape(-1, 300)) 
    for i in range(len(sentences_train))
]

labels_train[np.argmax(scores)]

#%% ENTITY EXTRACTION
import spacy

nlp = spacy.load('en_core_web_lg')

doc = nlp("my friend Mary has worked at Google since 2009")

for ent in doc.ents:
    print(ent.text, ent.label_)

# Dependency parsing
doc = nlp("a flight to Shanghai from Singapore")
shanghai, singapore = doc[3], doc[5]

print(list(shanghai.ancestors)) # [to, flight]
print(list(singapore.ancestors)) # [from, flight]

doc = nlp("let's see that jacket in red and some blue jeans")
items = [doc[4], doc[10]] # [jacket, jeans]
colors = [doc[6], doc[9]] # [red, blue]

for color in colors:
    for token in color.ancestors:
        if token in items:
            print(f"color {color} belongs to item {token}")
            break

# %% ROBUST NLU WITH RASA: RASA NLU python library
# - Library for intent recognition and entity extraction
# - Based on spaCy, scikit-learn and other libraries
# - Built in support for chatbot specific tasks