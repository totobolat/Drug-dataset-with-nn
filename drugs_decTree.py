import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df = pd.read_csv('./input/drug200.csv')
print(df.head())

strCol = df.iloc[:, [1,2,3,5]]
intCol = df.iloc[:,[0, 4]]
a_enc = strCol.copy()
for col in strCol.columns:
    lb = LabelEncoder()
    a_enc[col] = lb.fit_transform(strCol[col].values)


dfenc = pd.concat([intCol, a_enc], axis = 1)
print(dfenc.head(10))

train_dataset = dfenc.sample(frac=0.8, random_state=0)
test_dataset = dfenc.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Drug')
test_labels = test_features.pop('Drug')

model = Sequential(
    [
        Dense(units=128, activation=tf.keras.activations.relu),
        Dense(units=64, activation=tf.keras.activations.relu),
        Dense(units=32, activation=tf.keras.activations.relu),
        Dense(units=5, activation=tf.keras.activations.softmax)
    ]
)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=500,verbose=2)

model.evaluate(test_features, test_labels)
#relu : loss: 0.0729 - accuracy: 0.9750
#swish : loss: 0.0496 - accuracy: 1.0000
#gauss : loss: 0.0410 - accuracy: 1.0000
#tanh : loss: 0.0246 - accuracy: 1.0000
#selu : loss: 0.0242 - accuracy: 1.0000
#softplus : loss: 0.0151 - accuracy: 1.0000
#again swish : loss: 0.0077 - accuracy: 1.0000
#again relu : loss: 0.0490 - accuracy: 0.9750