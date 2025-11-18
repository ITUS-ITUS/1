import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

bias = 1
l = 0.1

# Input patterns with bias
x_data = np.array([
    [-1,-1,bias],
    [-1,1,bias],
    [1,-1,bias],
    [1,1,bias]
])

# Target outputs
y_and     = np.array([-1,-1,-1,1])
y_or      = np.array([-1,1,1,1])
y_andnot  = np.array([-1,-1,1,-1])
y_xor     = np.array([-1,1,1,-1])

# Hebbian training
def hebb_train(X, Y, lr):
    w = np.zeros(3)
    for i in range(len(X)):
        w += X[i] * Y[i] * lr
        print(f"Update {i+1}: w = {w}")
    print("Final weights:", w)
    return w

# Prediction using your original function
def predict(x, w):
    x_bias = np.append(x, 1)
    result = np.dot(x_bias, w)
    return 1 if result >= 0 else -1

# Wrapper to predict all inputs
def predict_all(X, w):
    return np.array([predict(x[:2], w) for x in X])

# Training and evaluation for all gates
gates = {
    "AND": y_and,
    "OR": y_or,
    "ANDNOT": y_andnot,
    "XOR": y_xor
}

for name, labels in gates.items():
    print("\n==============================")
    print("Training for", name)
    print("==============================")

    w_final = hebb_train(x_data, labels, l)

    predictions = predict_all(x_data, w_final)

    # Accuracy
    acc = accuracy_score(labels, predictions) * 100
    print("Accuracy:", acc, "%")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[-1, 1])
    print("\nConfusion Matrix:")
    print(cm)
