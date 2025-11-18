import numpy as np 

data = np.array(
    [[0,0],
     [0,1],
     [1,1],
     [1,0]]
)

try:
    def is_and(x,y):
        w1=w2=1
    
        if (w1*x)+ (w2*y) > 1:
            return True
        else:
            return False
    
    def is_or(x,y):
        w1=w2=1

        if (w1*x)+(w2*y) >= 1:
            return True
        else:
            return False
    
    def is_xor(x,y):
        w1=w2=1
        
        if(w1*x) +(w2*y) ==1:
            return True
        else:
            return False

except Exception as e:
    print(e)

else:
    print("and")
    for (x,y) in data:
        and_result = is_and(x,y)
        print(f"input value is {x}, {y} simualtion AND is {and_result}")

    print("or")
    for (x,y) in data:
        or_result = is_or(x,y)
        print(f"input values is {x}, {y} simulation OR is {or_result}")

    print("xor")
    for (x,y) in data:
        xor_result = is_xor(x,y)
        print(f"input values is {x}, {y} simulation XOR is {xor_result}")

finally:
    print("simultion done!!!!")