import numpy as np

print("MADALINE Neural Network")
print()

X_train = [
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1]
]

y_train = [1, -1, -1, 1]

print("Training XOR with MADALINE:")
for i in range(len(X_train)):
    print(f"Input: {X_train[i]} -> Target: {y_train[i]}")

hidden_weights1 = [0.1, 0.2, -0.3]
hidden_weights2 = [-0.2, 0.3, 0.1]
output_weights = [0.1, 0.4, -0.2]

learning_rate = 0.1
epochs = 2000

print()
print("Training MADALINE network...")

for epoch in range(epochs):
    total_error = 0
    
    for i in range(len(X_train)):
        x = X_train[i]
        target = y_train[i]
        
        h1_net = 0
        for j in range(3):
            h1_net += hidden_weights1[j] * x[j]
        h1_out = 1 if h1_net >= 0 else -1
        
        h2_net = 0
        for j in range(3):
            h2_net += hidden_weights2[j] * x[j]
        h2_out = 1 if h2_net >= 0 else -1
        
        output_input = [1, h1_out, h2_out]
        output_net = 0
        for j in range(3):
            output_net += output_weights[j] * output_input[j]
        output = 1 if output_net >= 0 else -1
        
        error = target - output
        total_error += abs(error)
        
        if error != 0:
            for j in range(3):
                output_weights[j] = output_weights[j] + learning_rate * error * output_input[j]
            
            if h1_out == target and h2_out != target:
                for j in range(3):
                    hidden_weights2[j] = hidden_weights2[j] + learning_rate * target * x[j]
            elif h1_out != target and h2_out == target:
                for j in range(3):
                    hidden_weights1[j] = hidden_weights1[j] + learning_rate * target * x[j]
            elif h1_out != target and h2_out != target:
                if abs(h1_net) < abs(h2_net):
                    for j in range(3):
                        hidden_weights1[j] = hidden_weights1[j] + learning_rate * target * x[j]
                else:
                    for j in range(3):
                        hidden_weights2[j] = hidden_weights2[j] + learning_rate * target * x[j]
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Total Error = {total_error}")

print()
print("Final weights:")
print(f"Hidden unit 1: {hidden_weights1}")
print(f"Hidden unit 2: {hidden_weights2}")  
print(f"Output unit: {output_weights}")

print()
print("Testing MADALINE:")
print("Input\t\tH1\tH2\tOutput\tTarget")
print("-" * 40)

for i in range(len(X_train)):
    x = X_train[i]
    target = y_train[i]
    
    h1_net = 0
    for j in range(3):
        h1_net += hidden_weights1[j] * x[j]
    h1_out = 1 if h1_net >= 0 else -1
    
    h2_net = 0
    for j in range(3):
        h2_net += hidden_weights2[j] * x[j]
    h2_out = 1 if h2_net >= 0 else -1
    
    output_input = [1, h1_out, h2_out]
    output_net = 0
    for j in range(3):
        output_net += output_weights[j] * output_input[j]
    output = 1 if output_net >= 0 else -1
    
    print(f"{x}\t{h1_out}\t{h2_out}\t{output}\t{target}")

print()

accuracy = 0
for i in range(len(X_train)):
    x = X_train[i]
    target = y_train[i]
    
    h1_net = 0
    for j in range(3):
        h1_net += hidden_weights1[j] * x[j]
    h1_out = 1 if h1_net >= 0 else -1
    
    h2_net = 0
    for j in range(3):
        h2_net += hidden_weights2[j] * x[j]
    h2_out = 1 if h2_net >= 0 else -1
    
    output_input = [1, h1_out, h2_out]
    output_net = 0
    for j in range(3):
        output_net += output_weights[j] * output_input[j]
    output = 1 if output_net >= 0 else -1
    
    if output == target:
        accuracy += 1

print(f"Accuracy: {accuracy}/{len(X_train)} = {accuracy/len(X_train)*100:.1f}%")
print()
print("MADALINE training completed!")