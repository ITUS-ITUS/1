import numpy as np
import math

print("Backpropagation Neural Network - Single Hidden Layer")
print()

X = [
    [0, 0],
    [0, 1], 
    [1, 0],
    [1, 1]
]

y = [0, 1, 1, 0]

print("Training XOR with Backpropagation:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Target: {y[i]}")

input_size = 2
hidden_size = 3
output_size = 1

W1 = [
    [0.5, -0.2, 0.3],
    [0.1, 0.4, -0.1]
]
b1 = [0.1, -0.3, 0.2]

W2 = [0.2, -0.5, 0.4]
b2 = 0.1

learning_rate = 0.5
epochs = 5000

print()
print("Initial weights:")
print("W1 (input->hidden):", W1)
print("W2 (hidden->output):", W2)

def sigmoid(x):
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0
    return 1 / (1 + math.exp(-x))

print()
print("Training with backpropagation...")

for epoch in range(epochs):
    total_loss = 0
    
    for i in range(len(X)):
        x1, x2 = X[i]
        target = y[i]
        
        z1 = [0, 0, 0]
        a1 = [0, 0, 0]
        
        for j in range(3):
            z1[j] = W1[0][j] * x1 + W1[1][j] * x2 + b1[j]
            a1[j] = sigmoid(z1[j])
        
        z2 = W2[0] * a1[0] + W2[1] * a1[1] + W2[2] * a1[2] + b2
        a2 = sigmoid(z2)
        
        loss = (target - a2) ** 2
        total_loss += loss
        
        dL_da2 = -2 * (target - a2)
        da2_dz2 = a2 * (1 - a2)
        dL_dz2 = dL_da2 * da2_dz2
        
        dL_dW2 = [0, 0, 0]
        for j in range(3):
            dL_dW2[j] = dL_dz2 * a1[j]
        dL_db2 = dL_dz2
        
        dL_da1 = [0, 0, 0]
        for j in range(3):
            dL_da1[j] = dL_dz2 * W2[j]
        
        dL_dz1 = [0, 0, 0]
        for j in range(3):
            da1_dz1 = a1[j] * (1 - a1[j])
            dL_dz1[j] = dL_da1[j] * da1_dz1
        
        dL_dW1 = [[0, 0, 0], [0, 0, 0]]
        for j in range(3):
            dL_dW1[0][j] = dL_dz1[j] * x1
            dL_dW1[1][j] = dL_dz1[j] * x2
        
        dL_db1 = [0, 0, 0]
        for j in range(3):
            dL_db1[j] = dL_dz1[j]
        
        for j in range(3):
            W2[j] = W2[j] - learning_rate * dL_dW2[j]
        b2 = b2 - learning_rate * dL_db2
        
        for j in range(3):
            W1[0][j] = W1[0][j] - learning_rate * dL_dW1[0][j]
            W1[1][j] = W1[1][j] - learning_rate * dL_dW1[1][j]
            b1[j] = b1[j] - learning_rate * dL_db1[j]
    
    if epoch % 1000 == 0:
        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

print()
print("Training completed!")

print()
print("Final weights:")
print("W1:", W1)
print("W2:", W2)

print()
print("Testing trained network:")
print("Input\t\tOutput\t\tTarget\t\tError")
print("-" * 45)

for i in range(len(X)):
    x1, x2 = X[i]
    target = y[i]
    
    z1 = [0, 0, 0]
    a1 = [0, 0, 0]
    
    for j in range(3):
        z1[j] = W1[0][j] * x1 + W1[1][j] * x2 + b1[j]
        a1[j] = sigmoid(z1[j])
    
    z2 = W2[0] * a1[0] + W2[1] * a1[1] + W2[2] * a1[2] + b2
    output = sigmoid(z2)
    
    error = abs(target - output)
    prediction = 1 if output > 0.5 else 0
    
    print(f"{X[i]}\t\t{output:.4f} ({prediction})\t{target}\t\t{error:.4f}")

print()
print("Backpropagation training completed!")