import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data = np.random.rand(3, 100)
x, y, z = data  # for show
c = np.arange(len(x)) / len(x)  # create some colours
print(c.shape)
#c = np.arange(len(x)) / len(x)  # create some colours

p = ax.scatter(x, y, z, c=plt.cm.magma(c))
ax.set_xlabel('$\psi_1$')
ax.set_ylabel('$\Phi$')
ax.set_zlabel('$\psi_2$')

#ax.set_box_aspect([np.ptp(i) for i in data])  # equal aspect ratio

fig.colorbar(p, ax=ax)
plt.show()