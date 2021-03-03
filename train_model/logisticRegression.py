# -*- coding: utf-8 -*-

# importing libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# get current directory  
current_directory = os.getcwd()

# importing the dataset
dataset = pd.read_csv(current_directory+"/train_model/datasets/dataset.csv")
dataset = dataset.drop('id', 1)  # removing unwanted column
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# spliting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

# fitting logistic regression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train.ravel())

# predicting the tests set result
y_pred = classifier.predict(x_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy
print(accuracy_score(y_test, y_pred))

# pickle file joblib
joblib.dump(classifier, current_directory+"/final_models/logisticR_final2.pkl")
