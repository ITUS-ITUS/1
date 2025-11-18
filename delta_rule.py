import numpy as  np
try :
    x = np.array([
        [1,-2,1.5,0],
        [1,-0.5,-2,-1.5],
        [0,1,1,1.5]
    ])

    x = np.c_[np.ones(x.shape[0]),x]
    
    y_target = np.array([1,-1,0])

    weights = np.zeros(x.shape[1])
    lr = 0.1
    epochs = 100

    for e in range(epochs):
        for i in range(len(x)):
            xi = x[i]
            yi = xi @ weights

            error = y_target[i] - yi
            weights += xi * error * lr


        if e % 10 == 0:
            pride = x @ weights
            loss = np.mean((y_target - pride) ** 2)
            print(f"epoch {e} : loss is {loss}")

except Exception as e:
    print(e)

else:
    print(f"final weights is : {weights}")

finally:
    print("delta rule done !!")


# For regression tasks (predicting continuous outputs): Use Linear.

# For binary classification tasks: Use Sigmoid in the output layer.

# For multi-class classification tasks: Use Softmax in the output layer.

# For deep networks: ReLU or Leaky ReLU are common choices for hidden layers, though Swish is becoming increasingly popular.