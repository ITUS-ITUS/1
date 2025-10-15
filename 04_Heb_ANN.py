import numpy as np

print("Hebbian Artificial Neural Network")
print()

training_inputs = [
    [1, 1, -1],
    [1, -1, 1], 
    [-1, 1, 1],
    [-1, -1, -1]
]

training_outputs = [1, -1, -1, 1]

print("Training Data:")
for i in range(len(training_inputs)):
    print(f"Input: {training_inputs[i]} -> Output: {training_outputs[i]}")

weights = [0.0, 0.0, 0.0]

print(f"\nInitial weights: {weights}")

print("\nTraining with Hebbian rule...")
for i in range(len(training_inputs)):
    x = training_inputs[i]
    target = training_outputs[i]
    
    for j in range(3):
        weights[j] = weights[j] + x[j] * target
    
    print(f"After pattern {i+1}: weights = {weights}")

print(f"\nFinal weights: {weights}")

print("\nTesting the trained network:")
for i in range(len(training_inputs)):
    x = training_inputs[i]
    net = 0
    for j in range(3):
        net += weights[j] * x[j]
    
    output = 1 if net >= 0 else -1
    
    print(f"Input: {x} -> Net: {net:.2f} -> Output: {output}, Expected: {training_outputs[i]}")

print()

print("Testing with new patterns:")
test_patterns = [
    [1, 0, 1],
    [-1, 0, -1],
    [0, 1, 0]
]

for pattern in test_patterns:
    net = 0
    for j in range(3):
        net += weights[j] * pattern[j]
    
    output = 1 if net >= 0 else -1
    print(f"Input: {pattern} -> Net: {net:.2f} -> Output: {output}")

print()
print("Hebbian learning completed!")