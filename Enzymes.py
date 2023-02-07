import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pandas as pd
import numpy as np
import sklearn
import pickle
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt



from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

data = pd.read_csv("enz4.csv", encoding='cp1252')

d=data.isnull().sum()
le = LabelEncoder()
data["shape"] = le.fit_transform(data["shape"])

data["Surface modification "] = le.fit_transform(data["Surface modification "])
data["Dispersion medium"] = le.fit_transform(data["Dispersion medium"])
data["Bp"] = le.fit_transform(data["Bp"])
data["Substrate1 "] = le.fit_transform(data["Substrate1 "])
X = data.iloc[:,0:21].values
Z = data['Mimic enzyme activity']

le.fit(Z)
le.classes_
Y = le.transform(Z)
Y = to_categorical(Y)
epoches=10
batch_size=32

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
sc.fit(x_train)

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)



model = keras.Sequential([
    keras.layers.Dense(100, input_dim=21, kernel_initializer='he_uniform'),  # input layer (1)
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu'),  # hidden layer (2)
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),  # hidden layer (3)
    keras.layers.Dropout(0.2),
    keras.layers.Dense(5, activation='sigmoid') # output layer (4)
])



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train_scaled, y_train,validation_data=(x_test_scaled, y_test), epochs=epoches, batch_size=batch_size)


test_loss, test_acc = model.evaluate(x_test_scaled,  y_test, verbose=0)


y=model.predict(x_test_scaled)

num=np.random.randint(0, len(x_test))

print("predictions are : ", le.classes_[np.argmax(y[num])])
print("real values are ", le.classes_[np.argmax(y_test[num])])



y_pred=np.zeros(len(y))
y_orig=np.zeros(len(y))
for i in range (len(y_pred)):
    y_pred[i]=np.argmax(y[i])
    y_orig[i]=np.argmax(y_test[i])
print(y_pred)
print(y_orig)

print(accuracy_score(y_orig, y_pred))






