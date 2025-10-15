import numpy as np

print("AND Gate Training")
and_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
and_outputs = [0, 0, 0, 1]

w1 = 0.5
w2 = 0.5  
bias = -0.7
learning_rate = 0.1

for epoch in range(1000):
    for i in range(len(and_inputs)):
        x1, x2 = and_inputs[i]
        target = and_outputs[i]
        net_input = w1 * x1 + w2 * x2 + bias
        output = 1 if net_input >= 0 else 0
        error = target - output
        w1 = w1 + learning_rate * error * x1
        w2 = w2 + learning_rate * error * x2
        bias = bias + learning_rate * error

print("AND Gate Results:")
for i in range(len(and_inputs)):
    x1, x2 = and_inputs[i]
    net_input = w1 * x1 + w2 * x2 + bias
    output = 1 if net_input >= 0 else 0
    print(f"{x1} AND {x2} = {output}")

print()

print("OR Gate Training")

or_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
or_outputs = [0, 1, 1, 1]

w1 = 0.5
w2 = 0.5
bias = -0.3
learning_rate = 0.1

for epoch in range(1000):
    for i in range(len(or_inputs)):
        x1, x2 = or_inputs[i]
        target = or_outputs[i]
        
        net_input = w1 * x1 + w2 * x2 + bias
        output = 1 if net_input >= 0 else 0
        
        error = target - output
        w1 = w1 + learning_rate * error * x1
        w2 = w2 + learning_rate * error * x2
        bias = bias + learning_rate * error


print("OR Gate Results:")
for i in range(len(or_inputs)):
    x1, x2 = or_inputs[i]
    net_input = w1 * x1 + w2 * x2 + bias
    output = 1 if net_input >= 0 else 0
    print(f"{x1} OR {x2} = {output}")

print()

print("NOT Gate Training")

not_inputs = [[0], [1]]
not_outputs = [1, 0]

w1 = -0.5
bias = 0.3
learning_rate = 0.1

for epoch in range(1000):
    for i in range(len(not_inputs)):
        x1 = not_inputs[i][0]
        target = not_outputs[i]
        
        net_input = w1 * x1 + bias
        output = 1 if net_input >= 0 else 0
        
        error = target - output
        w1 = w1 + learning_rate * error * x1
        bias = bias + learning_rate * error

print("NOT Gate Results:")
for i in range(len(not_inputs)):
    x1 = not_inputs[i][0]
    net_input = w1 * x1 + bias
    output = 1 if net_input >= 0 else 0
    print(f"NOT {x1} = {output}")

print("\nAll gates trained successfully!")