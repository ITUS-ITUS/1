import numpy as np

# Training data
x = np.array([
    [1.0,  1.0,  1.0],
    [1.0, -1.0,  1.0],
    [-1.0, 1.0,  1.0],
    [-1.0, -1.0, -1.0]
])

t = np.array([1, 1, 1, -1])

# Initialize weights and bias
w = np.zeros(x.shape[1])
b = 0.0

eta = 0.1
epochs = 20
theta = 0.0

def bipolar_activation(y):
    return 1 if y >= theta else -1

# Training loop
for epoch in range(epochs):
    print(f"\nEPOCH {epoch + 1}")
    for xi, target in zip(x, t):

        # net input includes bias
        y_in = np.dot(w, xi) + b

        y = bipolar_activation(y_in)
        error = target - y

        # Update rule
        w = w + eta * error * xi
        b = b + eta * error

        print(f"input: {xi}, target: {target}, pred: {y}, "
              f"error: {error}, weights: {w}, bias: {b}")

# Final testing
print("\n===== FINAL TESTING =====")
for xi, target in zip(x, t):
    y_in = np.dot(w, xi) + b
    y = bipolar_activation(y_in)
    print(f"input: {xi}, target: {target}, predicted: {y}")
