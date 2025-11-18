import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Age': [21, 21, 23, 27, 22],
    'Gender': ['Female', 'Female', 'Male', 'Male', 'Male'],
    'Height': [1.62, 1.52, 1.8, 1.8, 1.78],
    'Weight': [64, 56, 77, 87, 89.8],
    'CALC': ['no', 'Sometimes', 'Frequently', 'Frequently', 'Sometimes'],
    'FAVC': ['no', 'no', 'no', 'no', 'no'],
    'FCVC': [2, 3, 2, 3, 2],
    'NCP': [3, 3, 3, 3, 1],
    'SCC': ['no', 'yes', 'no', 'no', 'no'],
    'SMOKE': ['no', 'yes', 'no', 'no', 'no'],
    'CH2O': [2, 3, 2, 2, 2],
    'family_history_with_overweight': ['yes', 'yes', 'yes', 'no', 'no'],
    'FAF': [0, 3, 2, 2, 0],
    'TUE': [1, 0, 1, 0, 0],
    'CAEC': ['Sometimes', 'Sometimes', 'Sometimes', 'Sometimes', 'Sometimes'],
    'MTRANS': ['Public_Transportation', 'Public_Transportation', 
               'Public_Transportation', 'Walking', 'Public_Transportation'],
    'NObeyesdad': ['Normal_Weight', 'Normal_Weight', 'Normal_Weight', 
                   'Overweight_Level_I', 'Overweight_Level_II']
}

df = pd.DataFrame(data)
print(df)

le = LabelEncoder()

categorical_columns = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    
x = df.drop('NObeyesdad', axis = 1)
y = df['NObeyesdad']

y_encoded = le.fit_transform(y)

print("Encoded Data : ", x)
print("Target Labels : ", y_encoded)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)

model = MLPClassifier(hidden_layer_sizes = (10, 5), max_iter = 1000, activation = 'relu', learning_rate_init = 0.01, random_state = 42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy * 100 :.2f}%")

