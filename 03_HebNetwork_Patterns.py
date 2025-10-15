import numpy as np

print("Hebbian Network for Alphabet Recognition")
print()

pattern_A = [
    0, 1, 0,
    1, 0, 1,
    1, 1, 1,
    1, 0, 1,
    1, 0, 1
]

pattern_B = [
    1, 1, 0,
    1, 0, 1,
    1, 1, 0,
    1, 0, 1,
    1, 1, 0
]

pattern_C = [
    0, 1, 1,
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
    0, 1, 1
]

for i in range(15):
    pattern_A[i] = 1 if pattern_A[i] == 1 else -1
    pattern_B[i] = 1 if pattern_B[i] == 1 else -1
    pattern_C[i] = 1 if pattern_C[i] == 1 else -1

patterns = [pattern_A, pattern_B, pattern_C]
names = ['A', 'B', 'C']

print("Training patterns:")
for i, name in enumerate(names):
    print(f"Pattern {name}:")
    for row in range(5):
        line = ""
        for col in range(3):
            pixel = patterns[i][row*3 + col]
            line += "X" if pixel == 1 else "_"
        print(line)
    print()

weights = []
for i in range(15):
    row = []
    for j in range(15):
        row.append(0.0)
    weights.append(row)

print("Training Hebbian network...")
for pattern in patterns:
    for i in range(15):
        for j in range(15):
            weights[i][j] += pattern[i] * pattern[j]

for i in range(15):
    weights[i][i] = 0

print("Training completed!")
print()

print("Testing with original patterns:")
for p, name in enumerate(names):
    pattern = patterns[p]
    
    output = []
    for i in range(15):
        sum_val = 0
        for j in range(15):
            sum_val += weights[i][j] * pattern[j]
        output.append(1 if sum_val >= 0 else -1)
    
    correct = True
    for i in range(15):
        if output[i] != pattern[i]:
            correct = False
            break
    
    print(f"Pattern {name}: {'Correct' if correct else 'Incorrect'}")
    
    print(f"Recalled {name}:")
    for row in range(5):
        line = ""
        for col in range(3):
            pixel = output[row*3 + col]
            line += "X" if pixel == 1 else "_"
        print(line)
    print()

print("Testing with noisy patterns:")
for p, name in enumerate(names):
    pattern = patterns[p].copy()
    
    noise_positions = [3, 7]
    for pos in noise_positions:
        pattern[pos] *= -1
    
    print(f"Noisy {name}:")
    for row in range(5):
        line = ""
        for col in range(3):
            pixel = pattern[row*3 + col]
            line += "X" if pixel == 1 else "_"
        print(line)
    
    output = []
    for i in range(15):
        sum_val = 0
        for j in range(15):
            sum_val += weights[i][j] * pattern[j]
        output.append(1 if sum_val >= 0 else -1)
    
    original = patterns[p]
    correct = True
    for i in range(15):
        if output[i] != original[i]:
            correct = False
            break
    
    print(f"Recalled from noisy {name}: {'Correct' if correct else 'Incorrect'}")
    
    print(f"Output:")
    for row in range(5):
        line = ""
        for col in range(3):
            pixel = output[row*3 + col]
            line += "X" if pixel == 1 else "_"
        print(line)
    print()

print("Pattern recognition completed!")