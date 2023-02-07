import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot
import shap
from sklearn.metrics import r2_score
import pickle

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data = pd.read_csv("enz4.csv", encoding='cp1252')

d=data.isnull().sum()
le = LabelEncoder()
ohe = OneHotEncoder()
data["shape"] = le.fit_transform(data["shape"])

data["Surface modification "] = le.fit_transform(data["Surface modification "])
data["Dispersion medium"] = le.fit_transform(data["Dispersion medium"])
data["Bp"] = le.fit_transform(data["Bp"])
data["Substrate1 "] = le.fit_transform(data["Substrate1 "])

data["Mimic enzyme activity"].unique()

addcol = pd.get_dummies(data["Mimic enzyme activity"])


data = data.join(addcol)

X = data.iloc[:,0:21].values
Y = data.loc[:,"oxidase"]


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
sc.fit(x_train)

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

kerasmodel= Sequential()
kerasmodel.add(Dense(100, input_dim=21))

kerasmodel.add(Dense(75, activation='relu'))
kerasmodel.add(Dense(1, activation='sigmoid'))




kerasmodel.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = kerasmodel.fit(x_train, y_train, epochs=100)



test_loss, test_acc = kerasmodel.evaluate(x_test,  y_test, verbose=1)

print('Test accuracy:', test_acc)

y=kerasmodel.predict(x_test)

print("predictions are : ", y[100:103].round())
print("real values are ", y_test[100:103])


# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'])


# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'])


pyplot.show()


r=r2_score(y_test,y)

print(y_test)
b=np.reshape(y,len(y))


y_train_pred=kerasmodel.predict(x_train)

r1=r2_score(y_train,y_train_pred)
r2=r2_score(y_test,y)

print('The Training R2: ',r1)
print('The Testing R2: ',r2)

explainer = shap.Explainer(kerasmodel.predict, x_test)
shap_values = explainer(x_test)
shap.summary_plot(shap_values)
