
                                                                                                                                
import numpy as np

# Delta rule
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])  # inputs
y = np.array([1, 1, -1, -1])  # targets (bipolar)

# Parameters
lr = 0.1
epochs = 20
weights = np.random.randn(X.shape[1])
bias = 0.0

for epoch in range(epochs):
    total_error = 0
    for xi, target in zip(X, y):
        # Net input
        net = np.dot(xi, weights) + bias
        # Linear output 
        output = net
        # Error
        error = target - output
        total_error += error**2

        # Delta rule update
        weights += lr * error * xi
        bias += lr * error

    print(f"Epoch {epoch+1}, Error: {total_error:.4f}")

print("Final weights:", weights)
print("Final bias:", bias)

                    
