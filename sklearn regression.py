import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
california=fetch_california_housing()
# Generate synthetic regression dataset
X = california.data          # 1000 samples, 2 features
y = california.target  # linear relation with noise
scaler=StandardScaler()
X=scaler.fit_transform(X)
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Define neural network
# hidden_layer_sizes = tuple -> each number = neurons in a hidden layer
mlp = MLPRegressor(hidden_layer_sizes=(32,16,8),  # 2 hidden layers: 10 neurons, 5 neurons
                   activation='tanh',           # activation function for hidden layers
                   solver='adam',               # optimizer (adam, sgd, lbfgs)
                   learning_rate_init=0.01,     # learning rate
                   max_iter=5000,               # epochs
                   random_state=42)
# Train
mlp.fit(X_train, y_train)
# Predict
y_pred = mlp.predict(X_test)
# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 score :",r2)
print("First 5 predictions:", y_pred[:5])
print("First 5 true values:", y_test[:5])
