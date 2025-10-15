
                                                                                                                                
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, epochs=5000):
        self.layers = layers
        self.lr = learning_rate
        self.epochs = epochs
        self.params = {}
        self._init_weights()

    def _init_weights(self):
        np.random.seed(42)
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] = np.random.randn(self.layers[i-1], self.layers[i]) * 0.1
            self.params[f"b{i}"] = np.zeros((1, self.layers[i]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        cache = {"A0": X}
        for i in range(1, len(self.layers)):
            Z = np.dot(cache[f"A{i-1}"], self.params[f"W{i}"]) + self.params[f"b{i}"]
            A = self.sigmoid(Z)
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = A
        return cache

    def backward(self, cache, y):
        grads = {}
        m = y.shape[0]
        L = len(self.layers) - 1

        # Output error
        A_L = cache[f"A{L}"]
        dZ = A_L - y
        grads[f"dW{L}"] = np.dot(cache[f"A{L-1}"].T, dZ) / m
        grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True) / m

        # Hidden layers
        for i in range(L-1, 0, -1):
            dA = np.dot(dZ, self.params[f"W{i+1}"].T)
            dZ = dA * self.sigmoid_derivative(cache[f"A{i}"])
            grads[f"dW{i}"] = np.dot(cache[f"A{i-1}"].T, dZ) / m
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads

    def update_params(self, grads):
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]

    def fit(self, X, y):
        for epoch in range(self.epochs):
            cache = self.forward(X)
            grads = self.backward(cache, y)
            self.update_params(grads)

            if (epoch+1) % 1000 == 0:
                loss = np.mean((y - cache[f"A{len(self.layers)-1}"])**2)
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        cache = self.forward(X)
        return (cache[f"A{len(self.layers)-1}"] > 0.5).astype(int)


# ---------------- Example ----------------
# New dataset (8 samples, 2 features, binary output)
X = np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[3,0],[3,1]])
y = np.array([[0],[0],[1],[1],[1],[1],[1],[1]])

# Neural network: 2 input → 3 hidden → 2 hidden → 1 output
nn = NeuralNetwork(layers=[2, 3, 2, 1], learning_rate=0.5, epochs=5000)
nn.fit(X, y)

# Predictions
print("\nPredictions:")
print(nn.predict(X).T)
print("Actual:")
print(y.T)

# Accuracy
accuracy = np.mean(nn.predict(X) == y)
print(f"Accuracy: {accuracy:.2f}")

                    
