import numpy as np

a = b = c = d = 0
for i in range(100):
    x, y, z, t = np.random.poisson((3, 4, 3, 2))
    a += x
    b += y
    c += z
    d += t

print(a/100)
print(b/100)
print(c/100)
print(d/100)
