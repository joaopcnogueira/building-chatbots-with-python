##### DESCRIPTION #####
# This script builds a Numpy Matrix (X) of 
# WordVec representations for the sentences in ATIS dataset
#######################

#%%
import numpy as np
import pandas as pd
import spacy
import os

this_file = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(this_file))
data_dir = os.path.join(base_dir, 'datasets')

# Load dataset
atis = pd.read_csv(os.path.join(data_dir, 'atis_intents.csv'))
labels = atis['intents'].tolist()
sentences = atis['sentences'].tolist()

# Load spacy model
nlp = spacy.load('en_core_web_lg')

# Calculate the length of sentences
n_sentences = len(sentences)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

# Initialize the array with zeros: X
X = np.zeros((n_sentences, embedding_dim))

# Iterate over the sentences
for idx, sentence in enumerate(sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X[idx, :] = doc.vector

# %%
