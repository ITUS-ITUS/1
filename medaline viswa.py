import numpy as np

class Medaline:
    def __init__(self,input_size=2,hidden_size=2,lr=0.1):
        self.w=np.array([[0.05,0.1],
                         [0.3,0.15]])
        self.b12=np.array([0.3,0.15])
        self.w2=np.array([0.5,0.5])
        self.b3=0.5

        self.lr=lr

    def sign(self,x):
        return 1 if x>=0 else -1
 
    def forward(self,x):
        net_hidden=np.dot(self.w,x)+self.b12
        h=np.array([self.sign(n) for n in net_hidden])

        net_out=np.dot(self.w2,h)+self.b3
        out=self.sign(net_out)

        return h,out

    def train(self,X,y,epochs=10):
        for epoch in range(epochs):
##            print(f"\n epoch {epoch+1}")
            for xi,target in zip(X,y):
                h,out=self.forward(xi)
##                print(f"Input: {xi}, Target: {target}, Pred: {out}. hidden: {h}")

                if out!=target:
                    for i in range(len(h)):
                        self.w[i]+=self.lr*(target-out)*xi
                        self.b12[i]+=self.lr*(target-out)
##                    self.b3+=self.lr*(target-out)

    def predict(self,X):
        preds=[]
        for xi in X:
            _,out=self.forward(xi)
            preds.append(out)
        return np.array(preds)

X=np.array([[-1,-1],
            [1,-1],
            [-1,1],
            [1,1]])
y=np.array([-1,-1,1,-1])

model=Medaline(lr=0.1)
model.train(X,y,epochs=5005)

print(f"\n final predictions: ",model.predict(X))
