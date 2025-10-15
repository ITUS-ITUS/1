import numpy as np
import math

print("Delta Rule Learning Rate Optimization")
print()

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
best_lr = 0
min_epochs = float('inf')

print("Testing different learning rates:")
print("Learning Rate\tEpochs\t\tFinal Error")
print("-" * 45)

for lr in learning_rates:
    w1, w2, bias = 0.1, 0.1, 0.1
    epochs = 0
    max_epochs = 10000
    
    for epoch in range(max_epochs):
        total_error = 0
        epochs += 1
        
        for i in range(len(X)):
            x1, x2 = X[i]
            target = y[i]
            
            net = w1 * x1 + w2 * x2 + bias
            output = 1 / (1 + math.exp(-net))
            
            error = target - output
            total_error += error ** 2
            
            derivative = output * (1 - output)
            
            w1 = w1 + lr * error * derivative * x1
            w2 = w2 + lr * error * derivative * x2
            bias = bias + lr * error * derivative
        
        if total_error < 0.01:
            break
    
    print(f"{lr:.2f}\t\t{epochs}\t\t{total_error:.6f}")
    
    if epochs < min_epochs:
        min_epochs = epochs
        best_lr = lr

print()
print(f"Optimal learning rate: {best_lr}")
print(f"Converged in: {min_epochs} epochs")

print()
print("Training with optimal learning rate:")

w1, w2, bias = 0.1, 0.1, 0.1
epochs = 0

for epoch in range(5000):
    total_error = 0
    epochs += 1
    
    for i in range(len(X)):
        x1, x2 = X[i]
        target = y[i]
        
        net = w1 * x1 + w2 * x2 + bias
        output = 1 / (1 + math.exp(-net))
        
        error = target - output
        total_error += error ** 2
        
        derivative = output * (1 - output)
        
        w1 = w1 + best_lr * error * derivative * x1
        w2 = w2 + best_lr * error * derivative * x2
        bias = bias + best_lr * error * derivative
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Error = {total_error:.6f}")
    
    if total_error < 0.01:
        break

print()
print(f"Final weights: w1={w1:.3f}, w2={w2:.3f}, bias={bias:.3f}")

print()
print("Testing final network:")
print("Input\t\tOutput\t\tTarget\t\tError")
print("-" * 45)

for i in range(len(X)):
    x1, x2 = X[i]
    target = y[i]
    
    net = w1 * x1 + w2 * x2 + bias
    output = 1 / (1 + math.exp(-net))
    error = abs(target - output)
    
    print(f"[{x1}, {x2}]\t\t{output:.4f}\t\t{target}\t\t{error:.4f}")

print()
print("Delta rule optimization completed!")