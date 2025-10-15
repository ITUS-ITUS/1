import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])


def is_and(x,y):
    w1=w2=1
   
    if (w1*x)+(w2*y)>1:
        return True
    else:
        return False


def is_or(x,y):
    w1=w2=1
   
    if (w1*x)+(w2*y)>=1:
        return True
    else:
        return False

def is_xor(x,y):
    w1=w2=1
   
    if (w1*x)+(w2*y)==1:
        return True
    else:
        return False





print("AND")
for (x,y) in X:
    result_and=is_and(x,y)
    print(f"Input ({x},{y}) AND output-> {result_and}")
print("-"*10)


    
print("OR")
for (x,y) in X:
     result_or=is_or(x,y)
     print(f"Input ({x},{y}) AND output-> {result_or}")
print("-"*10)

print("XOR")
for (x,y) in X:
    result_xor=is_xor(x,y)
    print(f"Input ({x},{y}) AND output-> {result_xor}")

