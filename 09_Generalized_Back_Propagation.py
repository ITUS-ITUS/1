import numpy as np
import math

print("Generalized Backpropagation Neural Network")
print()

X = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0], 
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]

y = [0, 1, 1, 0, 1, 0, 0, 1]

print("Training 3-bit parity with generalized backpropagation:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Target: {y[i]}")

layers = [3, 6, 4, 1]
num_layers = len(layers)

weights = []
biases = []

np.random.seed(123)
for i in range(num_layers - 1):
    w = []
    for j in range(layers[i]):
        row = []
        for k in range(layers[i+1]):
            row.append(np.random.uniform(-1, 1))
        w.append(row)
    weights.append(w)
    
    b = []
    for j in range(layers[i+1]):
        b.append(np.random.uniform(-1, 1))
    biases.append(b)

learning_rate = 0.3
epochs = 8000

print()
print(f"Network architecture: {' -> '.join(map(str, layers))}")
print("Training generalized backpropagation...")

def sigmoid(x):
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0
    return 1 / (1 + math.exp(-x))

for epoch in range(epochs):
    total_loss = 0
    
    for sample in range(len(X)):
        activations = []
        activations.append(X[sample])
        
        current_input = X[sample]
        for layer in range(num_layers - 1):
            next_layer = []
            for neuron in range(layers[layer + 1]):
                z = biases[layer][neuron]
                for prev_neuron in range(layers[layer]):
                    z += current_input[prev_neuron] * weights[layer][prev_neuron][neuron]
                a = sigmoid(z)
                next_layer.append(a)
            activations.append(next_layer)
            current_input = next_layer
        
        output = activations[-1][0]
        target = y[sample]
        loss = (target - output) ** 2
        total_loss += loss
        
        deltas = []
        
        output_error = target - output
        output_delta = output_error * output * (1 - output)
        deltas.append([output_delta])
        
        for layer in range(num_layers - 2, 0, -1):
            layer_deltas = []
            for neuron in range(layers[layer]):
                error = 0
                for next_neuron in range(layers[layer + 1]):
                    error += deltas[0][next_neuron] * weights[layer][neuron][next_neuron]
                delta = error * activations[layer][neuron] * (1 - activations[layer][neuron])
                layer_deltas.append(delta)
            deltas.insert(0, layer_deltas)
        
        for layer in range(num_layers - 1):
            for neuron in range(layers[layer]):
                for next_neuron in range(layers[layer + 1]):
                    gradient = deltas[layer][next_neuron] * activations[layer][neuron]
                    weights[layer][neuron][next_neuron] += learning_rate * gradient
            
            for next_neuron in range(layers[layer + 1]):
                bias_gradient = deltas[layer][next_neuron]
                biases[layer][next_neuron] += learning_rate * bias_gradient
    
    if epoch % 2000 == 0:
        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

print()
print("Training completed!")

print()
print("Testing generalized backpropagation network:")
print("Input\t\t\tOutput\t\tTarget\t\tCorrect?")
print("-" * 50)

correct_predictions = 0

for i in range(len(X)):
    current_input = X[i]
    
    for layer in range(num_layers - 1):
        next_layer = []
        for neuron in range(layers[layer + 1]):
            z = biases[layer][neuron]
            for prev_neuron in range(layers[layer]):
                z += current_input[prev_neuron] * weights[layer][prev_neuron][neuron]
            a = sigmoid(z)
            next_layer.append(a)
        current_input = next_layer
    
    output_val = current_input[0]
    target_val = y[i]
    prediction = 1 if output_val > 0.5 else 0
    is_correct = prediction == target_val
    
    if is_correct:
        correct_predictions += 1
        
    print(f"{X[i]}\t{output_val:.4f} ({prediction})\t{target_val}\t\t{is_correct}")

accuracy = correct_predictions / len(X) * 100
print()
print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(X)})")

print()
print(f"Network has {num_layers-1} weight matrices")
for i in range(len(weights)):
    print(f"Layer {i+1} weights shape: {len(weights[i])}x{len(weights[i][0])}")

print()
print("Generalized backpropagation completed!")