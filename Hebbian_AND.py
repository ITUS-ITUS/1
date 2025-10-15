import numpy as np
ini_weights=np.array([0,0,0])
bias=1
X = np.array([
    [-1, -1, bias],
    [-1,  1, bias],
    [ 1, -1, bias],
    [ 1,  1, bias]
])
Y = np.array([-1, 1, 1, 1])
def hebb(X,Y,weights):
    w=weights.reshape(3,1)
    Y_T=Y.reshape(4,1)
    w_new=w+np.dot(X.T,Y_T)
    print(w_new)
    return w_new


def predict(X,weights):
    input_vector=np.append(X,bias)
    result=np.dot(input_vector,weights)
    return 1 if result>=0 else -1

final_weights=hebb(X,Y,ini_weights)


test_vector=[1,1]
prediction=predict(test_vector,final_weights)
print(prediction)
import matplotlib.pyplot as plt


w1, w2, w3 = final_weights

x = np.linspace(-2, 2, 100)
y = -(w1 * x + w3) / w2

plt.plot(x, y, color='blue')  # Decision line

# Scatter points
for i in range(len(X)):
    color = 'green' if Y[i] == 1 else 'red'
    plt.scatter(X[i][0], X[i][1], color=color)

plt.show()
