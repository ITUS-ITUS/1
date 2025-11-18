import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("BACKPROPAGATION NEURAL NETWORK")
print("=" * 60)

# Training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

print("\nTraining Data (XOR Gate):")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Target: {y[i]}")

# ==========================================
# METHOD 1: Using sklearn
# ==========================================
print("\n" + "=" * 60)
print("METHOD 1: Backpropagation using sklearn")
print("=" * 60)

# Create neural network with backpropagation
model_sklearn = MLPClassifier(hidden_layer_sizes=(4,),
                              activation='relu',
                              solver='sgd',  # Stochastic Gradient Descent
                              learning_rate_init=0.1,
                              max_iter=5000,
                              random_state=42)

print("\nTraining...")
model_sklearn.fit(X, y)

# Predictions
predictions_sklearn = model_sklearn.predict(X)

print("\nResults:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions_sklearn[i]} (Target: {y[i]})")

accuracy_sklearn = np.mean(predictions_sklearn == y) * 100
print(f"\nAccuracy: {accuracy_sklearn:.2f}%")

# ==========================================
# METHOD 2: Using TensorFlow/Keras
# ==========================================
print("\n" + "=" * 60)
print("METHOD 2: Backpropagation using TensorFlow")
print("=" * 60)

# Create neural network model
model_tf = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model (backpropagation happens automatically)
model_tf.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

print("\nTraining...")
history = model_tf.fit(X, y, epochs=1000, verbose=0)

# Predictions
predictions_tf = (model_tf.predict(X) > 0.5).astype(int).flatten()

print("\nResults:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions_tf[i]} (Target: {y[i]})")

# Evaluate
loss, accuracy_tf = model_tf.evaluate(X, y, verbose=0)
print(f"\nAccuracy: {accuracy_tf * 100:.2f}%")
print(f"Loss: {loss:.4f}")

# ==========================================
# Testing with new data
# ==========================================
print("\n" + "=" * 60)
print("Testing with New Patterns")
print("=" * 60)

test_data = np.array([[0, 0], [1, 1]])

print("\nsklearn predictions:")
test_pred_sklearn = model_sklearn.predict(test_data)
for i in range(len(test_data)):
    print(f"Input: {test_data[i]} -> Predicted: {test_pred_sklearn[i]}")

print("\nTensorFlow predictions:")
test_pred_tf = (model_tf.predict(test_data, verbose=0) > 0.5).astype(int).flatten()
for i in range(len(test_data)):
    print(f"Input: {test_data[i]} -> Predicted: {test_pred_tf[i]}")

print("\n" + "=" * 60)
print("Backpropagation Completed!")
print("=" * 60)