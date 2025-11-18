import numpy as np
import matplotlib.pyplot as plt


t_pattern = np.array([[1,1,1],
                      [-1,1,-1],
                      [-1,1,-1]])

c_pattern = np.array([[1,1,1],
                      [1,-1,-1],
                      [1,1,1]])

inputs = np.array([t_pattern.flatten(),c_pattern.flatten()])
target = np.array([1,-1])

weight = np.zeros(t_pattern.flatten().shape)
bias = 0

for x,t in zip(inputs,target):
    weight += x*t
    bias += t

def pred(x):
    y_in = np.dot(weight,x)+bias
    return 1 if y_in >= 0 else -1

print("Predicted pattern T:",pred(t_pattern.flatten()))
print("Predicted pattern C:",pred(c_pattern.flatten()))

plt.imshow(t_pattern)
plt.show()
plt.imshow(c_pattern)
plt.show()
