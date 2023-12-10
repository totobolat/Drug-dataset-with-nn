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

# fig, ax = plt.subplots(figsize=(14, 7), layout='constrained')
# ax.set_xlabel('Cholesterol')
# ax.set_ylabel('Drug')
# ax.set_title('ch to Drug')
# plt.grid(True)
# ax.scatter(df['Cholesterol'],df['Drug'], c='red',marker='x')
# plt.plot(df['Drug'])
# plt.show()

# train_dataset = df.sample(frac=0.8, random_state=0)
# test_dataset = df.drop(train_dataset.index)
# train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataset, label="Drug")
# test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_dataset, label="Drug")

# model = tfdf.keras.RandomForestModel()
# model.fit(train_ds)
# model.compile(metrics=["accuracy"])
# print(model.evaluate(test_dataset))

#######################

# high_bp = []
# normal_bp = []
# low_bp = []
# for bp in df['BP']:
#     if(bp == 'HIGH'):
#         high_bp.append('1')
#         normal_bp.append('0')
#         low_bp.append('0')
#     elif(bp == 'NORMAL'):
#         high_bp.append('0')
#         normal_bp.append('1')
#         low_bp.append('0')
#     else :
#         high_bp.append('0')
#         normal_bp.append('0')
#         low_bp.append('1')

# df.insert(df.shape[1]-1,'High_bp',high_bp)
# df.insert(df.shape[1]-1,'Normal_bp',normal_bp)
# df.insert(df.shape[1]-1,'Low_bp',low_bp)
# df=df.drop(columns='BP',axis=1)


# drug_dict = {'drugA':[], 'drugB':[], 'drugC':[], 'drugX':[], 'drugY':[]}
# for drug in df['Drug']:
#     #for keys in range(len(drug_dict.keys())):
#     for keys in drug_dict:
#         if drug == keys:
#             drug_dict[keys].append(1)
#         else:
#             drug_dict[keys].append(0)
# drug_df = pd.DataFrame.from_dict(drug_dict)
# df=df.join(drug_df)
# df=df.drop(columns='Drug',axis=1)
# print(df.head())

######################################################

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