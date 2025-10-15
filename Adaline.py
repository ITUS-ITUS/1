
                                                                                                                                
import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        """Linear activation (identity function)."""
        return x

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            # Update weights using LMS rule
            self.weights += self.lr * X.T.dot(errors) / n_samples
            self.bias += self.lr * errors.mean()

            cost = (errors**2).sum() / (2.0 * n_samples)
            print(f"Epoch {epoch+1}/{self.epochs}, Cost: {cost:.4f}")

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# ---------------- Example ----------------

# Dataset (X with 2 features, y = output)
X = np.array([
    [-5, 2],
    [-3, 1],
    [-1, -2],
    [2, 0],
    [4, 3]
])
y = np.array([0, 0, 1, 1, 1])  # given outputs

# Train Adaline
adaline = Adaline(learning_rate=0.01, epochs=30)
adaline.fit(X, y)

# Predictions
print("\nPredictions:")
print(adaline.predict(X))
print("Actual:", y)

# Accuracy
accuracy = np.mean(adaline.predict(X) == y)
print(f"Accuracy: {accuracy:.2f}")

                        
                    
