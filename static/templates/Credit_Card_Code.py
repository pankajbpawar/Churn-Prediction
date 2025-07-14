import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow

df1 = pd.read_csv('Churn.csv')
df1.head()

df = df1.drop(['RowNumber','CustomerId','Surname','Geography'] , axis = 1)
df.rename(columns={"Exited": "Allowance"}, inplace=True)
df.head()

g_mapping = {'Female' : 0 , 'Male' : 1}
df['Gender'] = df['Gender'].map(g_mapping)

plt.figure(figsize=(12, 8))  
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

x = df.drop('Allowance' , axis = 1)
y = df['Allowance']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  

from tensorflow import keras

credit_card = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  
])

credit_card.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

credit_card.fit(x_train, y_train, epochs=50, batch_size=34)

loss, accuracy = credit_card.evaluate(x_test, y_test)
print("Loss :" , loss)
print("Accuracy :" , accuracy)

import pickle

with open('credit_card_pickle', 'wb') as f:
    pickle.dump(credit_card,f)