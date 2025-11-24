import numpy as np
import simulation_alt as obj
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pso import Particle_Swarm_Optimization
import timeit
from utils import rotatePoint, relative_error
import pandas as pd

#Simulador UV-C
necessary_dosage = 16.9 #mJ/cm^2
power = 60 #W
attenuation = 10 #%
exposure_time = 0 #segundos
file_name = 'mapa1.txt' #Arquivo com o mapa
initial_pos = np.array([5, 6]) #Ponto de partida
robot_dim = 2 #Dimensões do robô
'''
#PSO
n_particles = 100
n_dimensions = 3
c1 = 2.05
c2 = 2.05
w_initial = 0.9
w_final = 0.1
n_iterations = 50
min_bound = -30
max_bound =  30
bounds = np.array([min_bound, max_bound])
'''
#PSO
n_particles = 100
n_dimensions = 3
c1 = 2.05
c2 = 2.05
w_initial = 0.9
w_final = 0.1
n_iterations = 100
min_bound = -10
max_bound =  10
bounds = np.array([min_bound, max_bound])

#Input
delta_input = np.array([3, 3, 3])
delta1 = np.array([2, 2, 5])

class ScanMatching():
    def __init__(self, optimizer, solution, bound=30):
        self.ref_scan = None
        self.sec_scan = None
        self.ref_raster = None
        self.sec_raster = None
        self.raster_shape = None

        #Parameters
        self.optimizer = optimizer
        self.input_solution = solution
        self.bound = bound

        #Results
        self.bestfitness = None
        self.solution = None
        self.solution_raster = None
        self.error = None

        #Animation
        self.df = None
        self.graph = None
        self.data = None
    
    def load_scans(self, ref_scan, sec_scan, raster_shape=(200, 200)):
        self.raster_shape = raster_shape
        self.ref_scan = ref_scan
        self.sec_scan = sec_scan
        self.ref_raster = self.read_scanning()
        self.sec_raster = self.read_scanning(use_sec_scan=True)

    def read_scanning(self, delta=np.zeros(3), use_sec_scan=False):
        scan = self.ref_scan
        if(use_sec_scan):
            scan = self.sec_scan
        
        raster = np.zeros(shape=self.raster_shape)
        i_center, j_center = (self.raster_shape[0]-1)/2, (self.raster_shape[1]-1)/2
        depth = 1
        for angle, distance in enumerate(scan):
            if(distance < 95):
                radians = angle*np.pi/180
                j = i_center + distance*np.math.sin(radians)
                i = j_center + distance*np.math.cos(radians)
                
                if(np.any(delta)):
                    x, y, theta = -delta[1], -delta[0], -delta[2]
                    (i, j) = rotatePoint([i_center + x, j_center+y], [i + x, j + y], theta) #Cuidado

                i, j = round(i), round(j)
                for di in range(-depth, depth + 1):
                    for dj in range(-depth, depth + 1):
                        if((i + di) >= 0 and (i + di) < self.raster_shape[0]):
                            if((j + dj) >= 0 and (j + dj) < self.raster_shape[1]):
                                #if(raster[i+di][j+dj] != 1):
                                raster[i+di][j+dj] += 1/(1+abs(di)+abs(dj))
                                #if(raster[i+di][j+dj] > 1):
                                    #raster[i+di][j+dj] = 1
        
        raster = raster/raster.max()
        return raster

    def DiceScore(self, ref_raster, new_raster, add = 0):
        x, y = ref_raster, new_raster
        score = (2*(x*y).sum() + 1e-6)/((x+y).sum() + 1e-6)
        return 1-(score + 0.1*add)

    def compare_raster(self, solution):
        cost = 0
        aux = np.array(3*[self.bound])
        #new_raster = self.read_scanning(delta = solution*aux)
        #add = np.median(abs(solution)%1)
        #cost += self.DiceScore(self.sec_raster, new_raster)
        
        new_raster = self.read_scanning(delta = -solution*aux, use_sec_scan=True)
        #add = np.median(abs(solution)%1)
        cost += self.DiceScore(self.ref_raster, new_raster)
        cost /= 2
        
        return cost

    def run(self):
        self.bestfitness, self.solution = self.optimizer.optimize(self.compare_raster, attractive_repulsive = False)
        self.solution *= np.array(3*[self.bound])
        print("Melhor solução:", self.solution, "Bestfitness:", self.bestfitness[-1])
        self.solution_raster = self.read_scanning(delta = -self.solution, use_sec_scan=True)#self.read_scanning(delta = self.solution)

        global bounds
        self.error = relative_error(self.input_solution, self.solution, bounds=bounds)
        print("Erro relativo:", self.error*100, "%")

    def plot(self):
        plt.title('Best fitness')
        plt.plot(self.bestfitness)
        plt.show()
        plt.imshow(self.ref_raster + self.sec_raster, cmap='gray_r')
        plt.show()
        plt.imshow(self.ref_raster + self.solution_raster, cmap='gray_r')
        plt.show()

    def print_file(self):
        pass
  
def main(input_solution):
    simulation = obj.simulation()
    simulation.create_display(file_name, initial_pos, robot_dim)
    simulation.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='scanning',delta = delta1, pso=True)
    distances1 = simulation.distances
    simulation.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='scanning', delta = input_solution, pso=True)
    distances2 = simulation.distances
    pygame.quit()

    print("Entrada:", input_solution)
    pso = Particle_Swarm_Optimization(n_particles, n_dimensions, c1, c2, w_initial, w_final, n_iterations)
    scan_matching = ScanMatching(pso, input_solution)
    start = timeit.default_timer()
    scan_matching.load_scans(distances1, distances2)
    scan_matching.run()
    stop = timeit.default_timer()
    print("Tempo total (segundos):", (stop-start))
    scan_matching.plot()
    
    return scan_matching


if __name__ == '__main__':
    x = main(delta_input)

'''
a = x.optimizer.anim_input*30
t = x.optimizer.anim_time
df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})
#c = np.array([np.mean(relative_error(delta_input, i, bounds)) for i in a[:n_particles]])
#c = lambda num : np.array([np.mean(relative_error(delta_input, i, bounds)) for i in a[num*n_particles:(num+1)*n_particles]])

def update_graph(num):
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    title.set_text('3D Test, time={}'.format(num))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

data=df[df['time']==0]
graph = ax.scatter(data.x, data.y, data.z)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$\theta$')
#graph = ax.scatter(data.x, data.y, data.z, c=plt.cm.magma(c))
fig.colorbar(graph, ax=ax)
ani = FuncAnimation(fig, update_graph, n_iterations-1, 
                               interval=80, blit=False)
ani.save('pso_convergence.gif', writer='ffmpeg')
plt.show()'''