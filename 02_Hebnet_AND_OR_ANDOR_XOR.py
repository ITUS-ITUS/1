import numpy as np

print("Hebbian Learning for Logic Gates")
print()

print("AND Gate:")
and_x = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
and_y = [1, -1, -1, -1]
w = [0, 0, 0]

for i in range(4):
    x1, x2 = and_x[i]
    target = and_y[i]
    w[0] += x1 * target
    w[1] += x2 * target
    w[2] += 1 * target

for i in range(4):
    x1, x2 = and_x[i]
    result = w[0]*x1 + w[1]*x2 + w[2]
    output = 1 if result >= 0 else 0
    input1 = 1 if x1 == 1 else 0
    input2 = 1 if x2 == 1 else 0
    print(f"{input1} AND {input2} = {output}")

print()

print("OR Gate:")
or_x = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
or_y = [1, 1, 1, -1]
w = [0, 0, 0]

for i in range(4):
    x1, x2 = or_x[i]
    target = or_y[i]
    w[0] += x1 * target
    w[1] += x2 * target
    w[2] += 1 * target

for i in range(4):
    x1, x2 = or_x[i]
    result = w[0]*x1 + w[1]*x2 + w[2]
    output = 1 if result >= 0 else 0
    input1 = 1 if x1 == 1 else 0
    input2 = 1 if x2 == 1 else 0
    print(f"{input1} OR {input2} = {output}")

print()

print("NOT Gate:")
not_x = [1, -1]
not_y = [-1, 1]
w = [0, 0]

for i in range(2):
    x1 = not_x[i]
    target = not_y[i]
    w[0] += x1 * target
    w[1] += 1 * target

for i in range(2):
    x1 = not_x[i]
    result = w[0]*x1 + w[1]
    output = 1 if result >= 0 else 0
    input1 = 1 if x1 == 1 else 0
    print(f"NOT {input1} = {output}")

print()

print("XOR Gate:")
xor_x = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
xor_y = [-1, 1, 1, -1]
w = [0, 0, 0]

for i in range(4):
    x1, x2 = xor_x[i]
    target = xor_y[i]
    w[0] += x1 * target
    w[1] += x2 * target
    w[2] += 1 * target

for i in range(4):
    x1, x2 = xor_x[i]
    result = w[0]*x1 + w[1]*x2 + w[2]
    output = 1 if result >= 0 else 0
    input1 = 1 if x1 == 1 else 0
    input2 = 1 if x2 == 1 else 0
    expected = 1 if xor_y[i] == 1 else 0
    print(f"{input1} XOR {input2} = {output} (expected {expected})")

print()
print("XOR fails with Hebbian learning")