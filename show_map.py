import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import colors

plt.close('all')

#Ler arquivo e armazenar em vetor
f = open('mapa0.txt') # opening a file
lines = f.read() # reading a file
lines = [x for x in lines if (x != '\n')]
mapa = np.array(list(lines))
#mapa = mapa.reshape(21, 36)
#mapa = mapa.reshape(13, 24)
mapa = mapa.reshape(12, 12)
mapa = mapa.astype(int)
f.close() # closing file object

fig_animate, ax = plt.subplots()

cmap = colors.ListedColormap(['Blue','red'])
#plt.figure(figsize=(6,6))
ax.pcolor(mapa[::-1], cmap=cmap,edgecolors='k', linewidths=3)

dots = []

dots.append(ax.plot([], [], linestyle='none', marker='s', markersize=35, color='teal'))

#ax.set_xlim([-1,11])
#ax.set_ylim([-1,11])

#Ler arquivo e armazenar em vetor
f = open('track.txt') # opening a file
lines = f.read() # reading a file
#lines = [x for x in lines_ if x != '\n']

lines_ = []
number = 0
k = 1
for x in lines:
    if((x == ',') or (x == '\n')):
        lines_.append(number)
        number = 0
        k = 1
    else:
        number = k*number + (int(x))
        k = 10*k
        
data = np.array(lines_)
data = data.astype(float)
data = data.reshape(data.size//2, 2)
data[:, 0] = 20 - data[:,0]
#data[:, 0] = 11 - data[:,0]
data[:, 1] = 1 + data[:,1]
f.close() # closing file object

def animate(z):
    data_y, data_x = data[z]
    dots[0][0].set_data(data_x, data_y)
    return dots

anim = animation.FuncAnimation(fig_animate, animate, frames=len(data), blit=False)

#ax.set_facecolor('#d3d3d3')
writer = animation.writers['ffmpeg'](fps=10)
dpi=500

anim.save('dot.mp4', writer=writer,dpi=dpi)