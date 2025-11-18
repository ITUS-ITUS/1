import numpy as np

X = np.array([[-1, -1 ,1],
              [-1, 1 ,1],
              [1, -1 ,1],
              [1, 1 ,1]])
y=np.array([-1,-1,-1,1])

w=np.array([0,0,0])
lr=0.5
for i in range(X.shape[0]):
    w+= X[i].T*y[i]

print(w)
print(np.where(np.dot(X,w)>=0,1,-1))
