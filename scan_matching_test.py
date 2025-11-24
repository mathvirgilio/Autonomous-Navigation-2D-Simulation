import numpy as np
import simulation_alt as obj
import pygame
from pso import Particle_Swarm_Optimization
import timeit
from scan_matching import ScanMatching
from numpy.random import randint, choice

#Simulador UV-C
necessary_dosage = 16.9 #mJ/cm^2
power = 60 #W
attenuation = 10 #%
exposure_time = 0 #segundos
file_name = 'mapa1.txt' #Arquivo com o mapa
initial_pos_list = [np.array([2, 2]), np.array([5, 2]), np.array([8, 2]), np.array([5, 5]), np.array([5, 8]), np.array([2, 8]), np.array([8, 8])] #Ponto de partida
robot_dim = 2 #Dimensões do robô
#PSO
#n_particles = [5, 10, 30, 50, 100]
#n_iterations = [25, 50, 100, 150, 200]
n_particles = 10
n_dimensions = 3
n_iterations = 100
c1 = 2.05
c2 = 2.05
w_initial = 0.9
w_final = 0.1
bound = 30
min_bound = -bound
max_bound =  bound
bounds = np.array([min_bound, max_bound])

f = open("scan_matching1.txt", "w")

def test(initial_pos, input_solution):
    simulation = obj.simulation()
    simulation.create_display(file_name, initial_pos, robot_dim)
    simulation.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='scanning', pso=True)
    distances1 = simulation.distances
    simulation.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='scanning', delta = input_solution, pso=True)
    distances2 = simulation.distances
    pygame.quit()

    print("Entrada:", input_solution)
    pso = Particle_Swarm_Optimization(n_particles, n_dimensions, c1, c2, w_initial, w_final, n_iterations)
    scan_matching = ScanMatching(pso, input_solution, bound=bound)
    start = timeit.default_timer()
    scan_matching.load_scans(distances1, distances2)
    scan_matching.run()
    stop = timeit.default_timer()
    total_time = stop-start
    print("Tempo total (segundos):", total_time)
    
    f.write(str(pso.n_particles) + ",")
    f.write(str(pso.n_iterations) + ",")
    f.write(str(initial_pos[0]) + ",")
    f.write(str(initial_pos[1]) + ",")
    f.write(str(scan_matching.input_solution[0]) + ",")
    f.write(str(scan_matching.input_solution[1]) + ",")
    f.write(str(scan_matching.input_solution[2]) + ",")
    f.write(str(scan_matching.solution[0]) + ",")
    f.write(str(scan_matching.solution[1]) + ",")
    f.write(str(scan_matching.solution[2]) + ",")
    f.write(str(scan_matching.bestfitness[-1]) + ",")
    f.write(str(total_time) + "\n")
    
    return scan_matching

f.write("Npart,Niter,posx,posy,Ix,Iy,Itheta,Ox,Oy,Otheta,f,t\n")
if __name__ == '__main__':
    for i in initial_pos_list:
        for j in range(9):
            delta_input = randint(low=20, high=30, size=3)*choice([-1,1], 3)
            for k in range(6):
                x = test(i, delta_input)
    f.close()