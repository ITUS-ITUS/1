import numpy as np
from sklearn.neural_network import MLPClassifier

print("MADALINE Neural Network")
print("=" * 50)

# Training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

print("\nTraining Data (XOR Gate):")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Target: {y[i]}")

# Create MADALINE (Multi-layer network)
madaline = MLPClassifier(hidden_layer_sizes=(4, 2),
                         max_iter=2000,
                         learning_rate_init=0.1,
                         random_state=42)

print("\nTraining MADALINE...")
madaline.fit(X, y)

print("Training completed!")
print(f"Layers: {madaline.n_layers_}")
print(f"Hidden units: {madaline.hidden_layer_sizes}")

# Testing
print("\n" + "=" * 50)
print("Testing Results:")
print("-" * 50)

predictions = madaline.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions[i]} (Target: {y[i]})")

# Accuracy
accuracy = np.mean(predictions == y) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Test new data
print("\n" + "=" * 50)
print("Testing New Patterns:")
test = np.array([[0, 0], [1, 1]])
test_pred = madaline.predict(test)
for i in range(len(test)):
    print(f"Input: {test[i]} -> Predicted: {test_pred[i]}")

print("\n" + "=" * 50)
print("MADALINE Completed!")







from sklearn.linear_model import Perceptron
import numpy as np

x = np.array([
    [0, 1],
    [0, -1],
    [1, 0],
    [1, 1]
])

y = np.array([1, -1, 1, 1])

for i in range(len(x)):
    print(f"Input : {x[i]} -> Target : {y[i]}")
    
model = Perceptron(max_iter = 1000, eta0 = 0.1, random_state = 42)
model.fit(x, y)

print(f"Final weights : {model.coef_[0]}")
print(f"Bias : {model.intercept_[0]}")

predictions = model.predict(x)

for i in range(len(x)):
    print(f"Input : {x[i]} -> Predicted : {predictions[i]:2d} (Target : {y[i]})")
    
accuracy = np.mean(predictions == y) * 100
print("Accuracy : ", accuracy)

test_data = np.array([
    [-1, -1],
    [-1, 1],
    [0, 0],
    [2, 2]
])

test_predictions = model.predict(test_data)
for i in range(len(test_data)):
    print(f"Input : {test_data[i]} -> Predicted Class : {test_predictions[i]:2d}")