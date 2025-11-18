import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(1)

# -------------------------
# Synthetic dataset (non-linear)
# -------------------------
N = 200
X = np.random.uniform(-2, 2, (N,2))  # 2 features

# non-linear but still learnable with linear hidden layer
y = 2*X[:,0] + 3*X[:,1] + 0.5*np.sin(X[:,0]*X[:,1]) + 0.1*np.random.randn(N)
y = y.reshape(-1,1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - X_mean)/X_std
X_test = (X_test - X_mean)/X_std

y_mean, y_std = y_train.mean(), y_train.std()
y_train = (y_train - y_mean)/y_std
y_test = (y_test - y_mean)/y_std

# -------------------------
# Your exact backprop code
# -------------------------
def backprop(X,y,lr,epochs,neurons):
    data,features = X.shape
    w1 = np.random.randn(features,neurons)
    b1 = np.zeros(neurons)
    w2 = np.random.randn(neurons,1)   
    b2 = 0

    for i in range(epochs):
        for j in range(X.shape[0]):
            xj = X[j].reshape(1,-1)
            yj = y[j].reshape(1,1)

            net1 = np.dot(xj,w1) + b1
            net2 = np.dot(net1,w2) + b2

            error = yj - net2

            # backprop
            delw2 = -2 * error * net1.T
            w2 -= lr * delw2

            delb2 = -2 * error
            b2 -= lr * delb2

            delb1 = -2 * error * w2.T
            delw1 = -2 * error * xj.T @ w2.T
            w1 -= lr * delw1
            b1 -= lr * delb1.flatten()

        if i % 500 == 0:
            preds = (np.dot(X,w1) + b1) @ w2 + b2
            loss = np.mean((y - preds)**2)
            print(f"Epoch {i}, Loss: {loss:.4f}")

        # NaN check
        if np.isnan(w1).any() or np.isnan(w2).any():
            print("NaN detected at epoch", i)
            break

    return w1,b1,w2,b2

# -------------------------
# Train
# -------------------------
w1,b1,w2,b2 = backprop(X_train, y_train, lr=0.001, epochs=3000, neurons=5)

# -------------------------
# Predict
# -------------------------
y_preds = (np.dot(X_test,w1) + b1) @ w2 + b2

print("\nR2 Score:", r2_score(y_test, y_preds))
print("Test MSE:", np.mean((y_test - y_preds)**2))
