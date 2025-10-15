import numpy as np

X=np.array([[1,1,1,1],
            [-1,1,-1,-1],
            [1,1,1,-1],
            [1,-1,-1,1]])
Y=np.array([1,1,-1,-1])
W=np.zeros(X.shape[1])
bias=0
alpha=0.5

first_error=99999
epochs=100

def AF(x):
    return x

for i in range(epochs):
    total_error=0
    for j in range(len(X)):
        xi=X[j]
        yi=Y[j]
        
        net=np.dot(W,xi)+bias
        ycap=AF(net)
        error=yi-ycap

        W=W+alpha*error*xi
        
        bias+=alpha*error
        total_error+=error**2
    
    avg_error=total_error/len(X)
    if (first_error-total_error)<0.001:
        print(f"Epoch : {i+1}     Total error : {total_error}")
        break;
    else:
        first_error=avg_error
    print(f"Epoch : {i+1}     AVG error : {total_error}")
        
    
    

        
        
