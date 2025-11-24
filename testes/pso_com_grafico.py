# Daniel Mauricio Muñoz Arboleda
# Algoritmo PSO basico
# Maio de 2009.
import numpy as np
import random as rdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import benchmarks as bm

#Animação dos pontos PSO
def animate(i):
    graph.set_data(x_plot[i], y_plot[i])
    return graph

#início (Funções para plot do gráfico da curva de convergência)
def data_gen(): #add t=0
    for i in range(maxiter):
        yield i, bestfitness[i]
    '''
    cnt = 0
    while cnt < maxiter:
        cnt += 1
        yield t, bestfitness[t]
        t += 1
    '''

def init():
    ax2.set_ylim(-100, 100)
    ax2.set_ylim(bottom=10**(-30))
    ax2.set_xlim(0, maxiter)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax2.get_xlim()
    if t >= xmax:
        ax2.set_xlim(xmin, 2*xmax)
        ax2.figure.canvas.draw()
        
    line.set_data(xdata, ydata)

    return line,
#fim

#Superfícies da animação do PSO
def cria_superficie(objfun, xg, yg):
    if objfun == 1:
        f = np.exp(-(xg-4)**2-(yg-4)**2)+np.exp(-(xg+4)**2-(yg-4)**2)+2*np.exp(-xg**2-(yg+4)**2)+2*np.exp(-xg**2-yg**2)
    elif objfun == 2:
        f = xg**2 + yg**2
    elif objfun == 3:
        f = xg**2+(xg+yg)**2    
    elif objfun == 4:
        f = 100*(yg+xg**2)+(1-xg)**2    
    elif objfun == 5:
        f=xg**2-10*np.cos(2*np.pi*xg)+10+yg**2-10*np.cos(2*np.pi*yg)+10
    elif objfun == 6:
        f = 418.9829*2-xg*np.sin(np.sqrt(abs(xg)))-yg*np.sin(np.sqrt(abs(yg)))
    elif objfun == 7:
        f = 20*np.exp(-0.2*np.sqrt(0.5*(xg**2+yg**2)))+20+np.exp(1)-np.exp(0.5*(np.cos(2*np.pi*xg)+np.cos(2*np.pi*yg)))
    elif objfun == 8:
        f = (1/4000)*(xg**2+yg**2)-np.cos(xg)*np.cos(yg/np.sqrt(2))+1
    else:
        f = xg**2+yg**2
    return f

## PSO PARAMETERS ##
S = 20 #number of particles
N = 2 #number of dimensions
maxiter = 200 # max number of iterations
w0 = 0.9 # initial weight
wf = 0.1 # final weight
slope = (wf-w0)/maxiter
w = np.array([w0+i*slope for i in range(maxiter)])
c1 = 2.05 # cognitive coefficient
c2 = 2.05 #1.9999#1.9999 # social coefficient
max_v = 3 # max velocity
ini_v = 0.5
# Range domain for fitness function
x_max = 30.0#8 #30 #500
x_min = -30.0 #-8 #-30 #-500
objfun = 2 #1:fourpeaks, 2:sphere, 3:quadric, 4:rosenbrock, 5:rastrigin, 6:schwefel, 7:ackley, 8:griewangk
threshold = 1e-10

x = np.zeros((S, N))
y = np.zeros((S, N))
v = np.zeros((S, N))

# INITIALIZATION
for i in range (S):
    for j in range(N):
        x[i, j] = x_min + (x_max-x_min)*rdm.uniform(0,1)
        y[i, j] = x_min + (x_max-x_min)*rdm.uniform(0,1)
        v[i, j] = ini_v

f_ind = 1e10*np.ones(maxiter) # initialize best fitness = 100

# Grid values are used for display only
Ngrid = 100
dx = (x_max - x_min)/Ngrid
xg, yg = np.meshgrid(np.arange(x_min, x_max, dx), np.arange(x_min, x_max, dx))
z = cria_superficie(objfun, xg, yg)

#Algorihtm parameters
f_ind = 1e10*np.ones(maxiter) # initialize best fitness = 100
fx = np.zeros(S)
bestfitness = np.zeros(maxiter)
#Animate plot elements
x_plot = {}
y_plot = {}

## ITERATIVE PROCESS
for k in range(maxiter):
    x_plot[k], y_plot[k] = np.copy(x).T #separa colunas de X para serem plotadas
    for i in range(S):
        if(objfun == 1):
            fx[i] = bm.sphere(x[i])
        elif(objfun == 2):
            fx[i] = bm.sphere(x[i])
        elif(objfun == 3):
            fx[i] = bm.quadric(x[i])    
        elif(objfun == 4):
            fx[i] = bm.rosenbrock(x[i])
        elif(objfun == 5):
            fx[i] = bm.rastrigin(x[i])
        elif(objfun == 6): 
            fx[i] = bm.schwefel(x[i])
        elif(objfun == 7): 
            fx[i] = bm.ackley(x[i])
        elif(objfun == 8):
            fx[i] = bm.griewank(x[i])
        else:
            fx[i] = bm.sphere(x[i])

        if (fx[i] < f_ind[i]):
            y[i] = x[i]
            f_ind[i] = fx[i]

    bestfitness[k] = np.amin(f_ind)
    p = np.argmin(f_ind)

    ys = y[p]
    
    # update particles
    for i in range(S):
        for j in range(N):
            r1 = rdm.uniform(0,1)
            r2 = rdm.uniform(0,1)
            v[i,j] = w[k]*v[i,j] + c1*r1*(y[i,j] - x[i,j]) + c2*r2*(ys[j] - x[i, j])

            if abs(v[i, j]) > max_v:
                if v[i, j] > 0:
                    v[i, j] = max_v
                else:
                    v[i, j] = -max_v
    
            x[i, j] = x[i, j] + v[i, j]
            if abs(x[i, j]) > x_max:
                if x[i, j] > 0:
                    x[i, j] = x_max
                else:
                    x[i, j] = -x_max


#Figure 1: Animating PSO
fig, ax = plt.subplots()
cp = ax.contourf(xg, yg, z, cmap=plt.cm.coolwarm)
fig.colorbar(cp)
ax.set_title('X1')
ax.set_ylabel('X2')
plt.xlim(-x_max, x_max)
plt.ylim(-x_max, x_max)


#PSO moving parts
graph, = ax.plot([], [], 'o', color='k')
anim = FuncAnimation(fig, animate, frames=maxiter, interval=200)
anim.save('pso.gif', writer='ffmpeg')
#plt.show()

#Best fitness
fig2, ax2 = plt.subplots()
ax2.set_title('Curva de convergência')
ax2.set_xlabel('Número de iterações')
line, = ax2.plot([], [], lw=2)
ax2.grid()
xdata, ydata = [], []
plt.yscale('log')

anim2 = FuncAnimation(fig2, run, data_gen, blit=False, interval=10,
                              repeat=False, init_func=init, save_count=200)
anim2.save('curva.gif', writer='ffmpeg')
#plt.show()