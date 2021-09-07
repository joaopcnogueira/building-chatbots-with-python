##### DESCRIPTION #####
# Trains a SVC to predict intents using word vectors as features

#%% Support Vector Classifier (SVC)
import numpy as np
import pandas as pd
import spacy
import os

this_file = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(this_file))
data_dir = os.path.join(base_dir, 'datasets')


# Load spacy model
nlp = spacy.load('en_core_web_lg')


# Load train dataset
atis_train = pd.read_csv(os.path.join(data_dir, 'atis_intents_train.csv'))
labels_train = atis_train['intents'].tolist()
sentences_train = atis_train['sentences'].tolist()

X_train_shape = (len(sentences_train), nlp.vocab.vectors_length)
X_train = np.zeros(X_train_shape)

for idx, sentence in enumerate(sentences_train):
    X_train[idx, :] = nlp(sentence).vector

y_train = np.array(labels_train)


# Load test dataset
atis_test = pd.read_csv(os.path.join(data_dir, 'atis_intents_test.csv'))
labels_test = atis_test['intents'].tolist()
sentences_test = atis_test['sentences'].tolist()

X_test_shape = (len(sentences_test), nlp.vocab.vectors_length)
X_test = np.zeros(X_test_shape)

for idx, sentence in enumerate(sentences_test):
    X_test[idx, :] = nlp(sentence).vector

y_test = np.array(labels_test)


# Create a support vector classifier
from sklearn.svm import SVC
clf = SVC(C=1)

# Fit the classifier using the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        n_correct += 1

print("Predicted {0} correctly out of {1} test examples".format(n_correct, len(y_test)))
# %%
