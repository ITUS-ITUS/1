import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_breast_cancer

breast_cancer=load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define neural network
mlp = MLPClassifier(hidden_layer_sizes=(32,16,8),   # 2 hidden layers of 5 neurons each
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.001,
                    max_iter=20000,
                    random_state=42)
# Train
mlp.fit(X_train, y_train)
# Predict
y_pred = mlp.predict(X_test)
# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

