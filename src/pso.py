import numpy as np
from numpy.random import rand, randint
import timeit
import matplotlib.animation
import pandas as pd

class Particle_Swarm_Optimization():
    def __init__(self, n_particles, n_dimensions, c1, c2, w_initial, w_final, n_iterations, min_bound=-1, max_bound=1, threshold = 0):
        #Parameters
        self.n_particles = n_particles # number of particles
        self.n_dimensions = n_dimensions # number of dimensions
        self.c1 = c1 # cognitive coefficient
        self.c2 = c2 # social coefficient
        self.n_iterations = n_iterations # max number of iterations
        self.w = np.array([w_initial+i*(w_final-w_initial)/n_iterations for i in range(n_iterations)]) # array of weight decaying
        self.bounds = {'min' : min_bound, 'max' : max_bound}
        self.threshold = threshold

        self.anim_input = np.empty((n_particles*n_iterations, n_dimensions))
        self.anim_time = np.array([np.ones(n_particles)*i for i in range(n_iterations)]).flatten()

    def optimize(self, cost_function, attractive_repulsive = False):

        #attractive and repulsive mode
        div_min = 0.1
        div_max = 1
        L = (self.n_particles*(self.bounds['max']-self.bounds['min'])**2)**(1/3) #Maior diagonal
        dir_ = 1

        ini_v = self.bounds['max']/20 #inicial velocity
        max_v = self.bounds['max']/5 #max velocity considering boundaries
        
        # INITIALIZATION
        x = np.empty(shape = (self.n_particles, self.n_dimensions), dtype = float)
        y = np.empty(shape = (self.n_particles, self.n_dimensions), dtype = float)
        v = np.empty(shape = (self.n_particles, self.n_dimensions), dtype = float)

        for i in range (self.n_particles):
            for j in range(self.n_dimensions):
                x[i, j] = self.bounds['min'] + (self.bounds['max']-self.bounds['min'])*rand()
                y[i, j] = self.bounds['min'] + (self.bounds['max']-self.bounds['min'])*rand()
                v[i, j] = ini_v
        
        #Algorihtm parameters
        f_ind = 1e10*np.ones(self.n_particles) # initialize best fitness = 100
        fx = np.zeros(self.n_particles)
        bestfitness = np.zeros(self.n_iterations)
        
        ## ITERATIVE PROCESS
        ITER = 0
        while (ITER < self.n_iterations):# and bestfitness[ITER] < self.threshold):

            self.anim_input[ITER*self.n_particles: (ITER+1)*self.n_particles] = x

            for i in range(self.n_particles):
                fx[i] = cost_function(x[i])
                if (fx[i] < f_ind[i]):
                    y[i] = x[i]
                    f_ind[i] = fx[i] #Melhor posição histórica individual
        
            bestfitness[ITER] = np.amin(f_ind) #Historical bestfitness
            p = np.argmin(f_ind)
            ys = y[p]
            
            # update particles
            for i in range(self.n_particles):
                for j in range(self.n_dimensions):
                    r1 = rand()
                    r2 = rand()
                    v[i,j] = self.w[ITER]*v[i,j] + dir_*self.c1*r1*(y[i,j] - x[i,j]) + dir_*self.c2*r2*(ys[j] - x[i, j])
        
                    if abs(v[i, j]) > max_v:
                        if v[i, j] > 0:
                            v[i, j] = max_v
                        else:
                            v[i, j] = -max_v
            
                    x[i, j] = x[i, j] + v[i, j]

                    if abs(x[i, j]) > self.bounds['max']:
                        if x[i, j] > 0:
                            x[i, j] = self.bounds['max']
                        else:
                            x[i, j] = self.bounds['min']
            

            #Calculo da diversidade
            if(attractive_repulsive):
                media_dim  = x.mean(axis=0)
                diversidade = np.sum((np.sum((x-media_dim)**2, axis=1))**(1/2))/self.n_particles/L
                print("Diversidade", diversidade)
                if(dir_ == 1 and diversidade < div_min):
                    dir_ = -1
                    div_min = div_min/2
                elif(dir_ == -1 and diversidade > div_max):
                    dir_ = 1

            #print("Iteração:", ITER + 1 ,"Melhor solução:", ys, "Bestfitness:", bestfitness[ITER])
            ITER += 1

        return bestfitness, ys