import numpy as np
import matplotlib.pyplot as plt

X=np.array([[1,1,1],
   [1,-1,1],
   [-1,1,1],
   [-1,-1,1]])

y=np.array([[1],
   [-1],
   [-1],
   [-1]])

w=np.array([[0],[0],[0]])

for i in range(4):
    wold=w
    w=wold+np.dot(np.transpose(X[i].reshape(1,3)),y[i].reshape(1,1))

x1_list=[float(i) for (i,j,k) in X]
x2_list=[float(j) for (i,j,k) in X]

m=-w[0]/w[1]
b=-w[2]/w[1]
plt.scatter(x1_list,x2_list,c=y)
plt.plot(np.arange(-7,7),[(m*i)+b for i in np.arange(-7,7)])
plt.show()
