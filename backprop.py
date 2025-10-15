import numpy as np

# ---------------- Activation functions ---------------- #
def relu(Z):
    return np.maximum(0, Z), Z

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear(Z):
    return Z, Z   # for regression output (identity)

def linear_backward_activation(dA, cache):
    return dA   # derivative of f(x)=x is 1

# ---------------- Initialization ---------------- #
def ini_paramters(layer_dim):
    parameters = {}
    L = len(layer_dim)
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dim[l], layer_dim[l-1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layer_dim[l], 1))
##    print(parameters)
    return parameters

# ---------------- Forward ---------------- #
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def L_layer_forward(X, parameters):
    A = X
    
    caches = []
    
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        caches.append((linear_cache, activation_cache, "relu"))

    # Last layer (linear)
    W = parameters["W"+str(L)]
    b = parameters["b"+str(L)]
    Z, linear_cache = linear_forward(A, W, b)
    A, activation_cache = linear(Z)
    caches.append((linear_cache, activation_cache, "linear"))

    return A, caches   # A = Y_hat

# ---------------- Loss ---------------- #
def compute_cost(Y_hat, Y):
    m = Y.shape[1]
    cost = (1/(2*m)) * np.sum((Y_hat - Y)**2)
    return cost

# ---------------- Backward ---------------- #
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def L_model_backward(Y_hat, Y, caches):
    grads = {}
    L = len(caches)
    m = Y.shape[1]

    dA = (Y_hat - Y)   # MSE derivative
    
    # Last layer (linear activation)
    linear_cache, activation_cache, activation = caches[-1]
    dZ = linear_backward_activation(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db
    dA = dA_prev

    # Hidden layers (ReLU)
    for l in reversed(range(L-1)):
        linear_cache, activation_cache, activation = caches[l]
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        grads["dW"+str(l+1)] = dW
        grads["db"+str(l+1)] = db
        dA = dA_prev

    return grads

# ---------------- Update ---------------- #
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters["W"+str(l)] -= learning_rate * grads["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate * grads["db"+str(l)]
    return parameters

# ---------------- Training ---------------- #
def model(X, Y, layer_dims, learning_rate=0.01, epochs=1000):
    parameters = ini_paramters(layer_dims)

    for i in range(epochs):
        Y_hat, caches = L_layer_forward(X, parameters)
        cost = compute_cost(Y_hat, Y)
        grads = L_model_backward(Y_hat, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print(f"Epoch {i}, Cost: {cost:.6f}")

    return parameters

# ---------------- Prediction ---------------- #
def predict(X, parameters):
    Y_hat, _ = L_layer_forward(X, parameters)
    return Y_hat

# ---------------- Test with dataset ---------------- #
X = np.array([
    [8, 85],   # Student 1: cgpa=8, result=85
    [7, 80],   # Student 2
    [9, 90],   # Student 3
    [6, 70]    # Student 4
]).T   # shape (2,4)

Y = np.array([
    [8, 7, 10, 5]
]).reshape(1, 4)   # shape (1,4)

layers = [2, 3, 2, 1]
param = model(X, Y, layers, learning_rate=0.001, epochs=1000)

print("\nPredictions:")
print(predict(X, param))
