import numpy as np
X=np.array([[1,-2,1.5,0,1],
            [1,-.5,-2,-1.5,1],
            [0,1,1,1.5,1]])
y=np.array([1,-1,0])
lr=0.01
w=np.random.randn(X.shape[1])
##w=np.array([1,-1,0,0.5])
for i in range(20000):
    for  j in range(X.shape[0]):
        net=np.dot(w.T,X[j])
        delw= lr*(y[j]-net)*X[j]
        w+=delw
print(w)
for i in range(X.shape[0]):
    print(f"for actual value {y[i]} : predicted value =>",round(np.dot(X[i],w),2))
