import numpy as np

try:
    x_data = np.array([
        [-1,-1,1],
        [-1,1,1],
        [1,1,1],
        [1,-1,1]
    ])

    y_and = np.array([-1,-1,-1,1])
    y_or = np.array([-1,1,1,1])
    y_andnot = np.array([-1,-1,1,-1])
    y_xor = np.array([-1,1,1,-1])

    l = 0.1
    w = np.zeros(3)

    def heb_net(x,y,w,r):
        for i in range(len(x)):
            new_w = w.T + (x[i].T * y[i] * r)
            wr = new_w
            print(wr)
        print(f"final W is :{new_w}")
    
except Exception as e:
    print(e)

else:
    print("and")
    heb_net(x_data,y_and,w,l)
    print("is")
    heb_net(x_data,y_or,w,l)
    print("andnot")
    heb_net(x_data,y_andnot,w,l)
    print("xor")
    heb_net(x_data,y_xor,w,l)

finally:
    print("Hebb net done !!")