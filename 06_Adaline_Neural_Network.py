import numpy as np

print("ADALINE Neural Network")
print()

X = [
    [1, 0, 1],
    [1, 0, -1],
    [1, 1, 0],
    [1, 1, 1]
]

y = [1, -1, 1, 1]

print("Training data:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Target: {y[i]}")

weights = [0.2, -0.3, 0.1]
learning_rate = 0.1
epochs = 1000

print(f"\nInitial weights: {weights}")
print(f"Learning rate: {learning_rate}")
print()

print("Training ADALINE...")

for epoch in range(epochs):
    total_error = 0
    
    for i in range(len(X)):
        x = X[i]
        target = y[i]
        
        output = 0
        for j in range(3):
            output += weights[j] * x[j]
        
        error = target - output
        total_error += error ** 2
        
        for j in range(3):
            weights[j] = weights[j] + learning_rate * error * x[j]
    
    if epoch % 100 == 0:
        mse = total_error / len(X)
        print(f"Epoch {epoch}: MSE = {mse:.6f}")

print()
print(f"Final weights: {weights}")

print()
print("Testing ADALINE:")
print("Input\t\t\tOutput\t\tTarget\t\tError")
print("-" * 50)

for i in range(len(X)):
    x = X[i]
    target = y[i]
    
    output = 0
    for j in range(3):
        output += weights[j] * x[j]
    
    error = target - output
    print(f"{x}\t{output:.3f}\t\t{target}\t\t{error:.3f}")

print()

print("Testing with new patterns:")
test_patterns = [
    [1, -1, -1],
    [1, -1, 1],
    [1, 0, 0]
]

for pattern in test_patterns:
    output = 0
    for j in range(3):
        output += weights[j] * pattern[j]
    
    classification = 1 if output >= 0 else -1
    print(f"Input: {pattern} -> Output: {output:.3f} -> Class: {classification}")

print()
print("ADALINE training completed!")