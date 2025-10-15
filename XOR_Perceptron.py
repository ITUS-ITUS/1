import numpy as np

def step(x):
    return np.where(x>=0,1,0)

class Perceptron:
    def __init__(self,input_dim,lr=0.1):
        self.weights=np.random.uniform(-1,1,input_dim)
        self.bias=0
        self.lr=lr

    def predict(self,x):
        return step(np.dot(x,self.weights)+self.bias)

    def train(self,X,y,epochs=10):
        for _ in range(epochs):
            for xi,target in zip(X,y):
                pred=self.predict(xi)

                error=target-pred
                self.weights+=self.lr*error*xi
                self.bias+=self.lr*error

X=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])
Y=np.array([[0],
            [1],
            [1],
            [0]])

p1=Perceptron(2,lr=0.1)
p2=Perceptron(2,lr=0.1)

Y_p1=np.array([0,0,1,0])
Y_p2=np.array([0,1,0,0])

p1.train(X,Y_p1,epochs=20)
p2.train(X,Y_p2,epochs=20)

p3=Perceptron(2,lr=0.1)
hidden_outputs=np.array([[p1.predict(x),p2.predict(x)]for x in X])
p3.train(hidden_outputs,Y.flatten(),epochs=20)

final_pred=[]
for x in X:
    h1=p1.predict(x)
    h2=p2.predict(x)

    out=p3.predict([h1,h2])
    final_pred.append(out)
print(final_pred)
