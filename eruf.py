import numpy as np 

x = np.array([
    [1,1,1],
    [1,-1,1],
    [-1,1,1],
    [-1,-1,-1]
])

t  = np.array([1,1,1,-1])

w = np.zeros(x.shape[0])

b = 0.0

lr = 0.1

epochs = 20

theta = 0.0

def bipolar_activation(t):
    return 1 if y  <= theta else -1

for i in range(epochs):
    print(f"epocha is { i+ 1}")
    for xi, target in zip(x,t):
        