
                                                                                                                                
import numpy as np

class Madaline:
    def __init__(self, n_hidden=3, learning_rate=0.1, epochs=10000):
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """Bipolar sigmoid (tanh)."""
        return np.tanh(x)

    def activation_deriv(self, x):
        """Derivative of tanh."""
        return 1 - np.tanh(x)**2

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.random.randn(n_features, self.n_hidden) * 0.1
        self.b = np.zeros((1, self.n_hidden))
        self.V = np.random.randn(self.n_hidden, 1) * 0.1
        self.bo = np.zeros((1, 1))

        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            # Forward
            hidden_input = np.dot(X, self.W) + self.b
            hidden_output = self.activation(hidden_input)

            final_input = np.dot(hidden_output, self.V) + self.bo
            output = self.activation(final_input)

            # Error
            error = y - output

            # Backprop
            d_output = error * self.activation_deriv(final_input)
            d_hidden = d_output.dot(self.V.T) * self.activation_deriv(hidden_input)

            # Update weights
            self.V += self.lr * hidden_output.T.dot(d_output)
            self.bo += self.lr * np.sum(d_output, axis=0, keepdims=True)
            self.W += self.lr * X.T.dot(d_hidden)
            self.b += self.lr * np.sum(d_hidden, axis=0, keepdims=True)

            # Stop if error is very small
            if np.mean(np.abs(error)) < 0.01:
                break

    def predict(self, X):
        hidden_output = self.activation(np.dot(X, self.W) + self.b)
        final_output = self.activation(np.dot(hidden_output, self.V) + self.bo)
        return np.where(final_output >= 0, 1, -1)


# ---------------- Example ----------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([-1, 1, 1, -1])   # Bipolar XOR

model = Madaline(n_hidden=3, learning_rate=0.1, epochs=10000)
model.fit(X, y)

print("\nPredictions:")
print(model.predict(X))
print("Actual:", y)

accuracy = np.mean(model.predict(X).T == y)
print(f"Accuracy: {accuracy:.2f}")

                    
