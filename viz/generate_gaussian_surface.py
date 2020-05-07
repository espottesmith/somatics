import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

a_s = [random.uniform(-50, -15) for e in range(32)]
xos = [random.uniform(0.05, 6.95) for e in range(32)]
yos = [random.uniform(0.05, 6.95) for e in range(32)]
sigxs = [random.uniform(0.25, 0.75) for e in range(32)]
sigys = [random.uniform(0.25, 0.75) for e in range(32)]

print(a_s)
print()
print(xos)
print()
print(yos)
print()
print(sigxs)
print()
print(sigys)

def mega_gaussian(x, y):
	z = np.zeros((1000, 1000))
	for i in range(32):
		z += a_s[i] * np.exp(-1 * ((x - xos[i])**2/(2 * sigxs[i]**2) + (y - yos[i])**2/(2 * sigys[i]**2)))
	return z

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x = np.linspace(0.0, 7.5, 1000)
y = np.linspace(0.0, 7.5, 1000)

xs, ys = np.meshgrid(x, y)
zs = mega_gaussian(xs, ys)

ax.plot_surface(xs, ys, zs, cmap="rainbow")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('E')

plt.show()