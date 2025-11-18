import numpy as np

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_derivation(x):
    return x * (1 - x)

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

for i in range(len(x)):
    print(f"Input : {x[i]} -> Target : {y[i][0]}")
    
np.random.seed(42)
weights = np.random.randn(2, 1)
bias = np.random.randn(1)
learning_rate = 0.5
epochs = 1000

print(f"Initial Weights : {weights.flatten()}")
print(f"Initial Bias : {bias[0]}")
print(f"Learning Rate : {learning_rate}")

for epoch in range(epochs):
    net_input = np.dot(x, weights) + bias
    output = sigmoid(net_input)
    
    error = y - output
    mse = np.mean(error ** 2)
    
    delta = error * sigmoid_derivation(output)
    
    weights += learning_rate * np.dot(x.T, delta)
    bias += learning_rate * np.sum(delta)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} : MSE = {mse:.6f}")
        
print(f"Final Weights : {weights.flatten()}")
print(f"Final Bias : {bias[0]}")

net_input = np.dot(x, weights) + bias
predictions = sigmoid(net_input)

for i in range(len(x)):
    prediction_class = 1 if predictions[i] >= 0.5 else 0
    print(f"Input : {x[i]} -> output : {predictions[i][0] : .4f} -> Class : {prediction_class}")
    
prediction_classes = (predictions >= 0.5).astype(int)
accuracy = np.mean(prediction_classes == y) * 100
print(f"Accuracy : {accuracy}")