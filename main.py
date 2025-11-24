import numpy as np
import sys
import timeit
import pygame
from src import simulation_alt as obj

#Informações UV-C
necessary_dosage = 16.9 #mJ/cm^2
power = 60 #60 #W
attenuation = 10 #%
exposure_time = 60 #segundos

file_name = 'mapas/mapa1.txt' #Arquivo com o mapa
initial_pos = np.array([2, 2]) #Ponto de partida
robot_dim = 2 #Dimensões do robô

start = timeit.default_timer()        
#Simulador
if __name__ == '__main__':
    simulation = obj.simulation()
    simulation.create_display(file_name, initial_pos, robot_dim)
    simulation.execute_navegation(power, necessary_dosage, attenuation, exposure_time)


stop = timeit.default_timer()
minutes = ((stop - start))*10//60
seconds= (stop - start)*10 - minutes*60
print('Tempo de execução do programa (em minutos): ', minutes, ":", seconds)

(x, y) = simulation.map.shape
aux = simulation.map.reshape((1, x*y))
count = aux[aux != '1'].shape[0]
area = count*(0.25*0.25)

(x, y) = simulation.dosage_per_block.shape
aux = simulation.dosage_per_block.reshape((1, x*y))
count_superior_dosage = aux[aux > necessary_dosage].shape[0]
total_dosage = aux.sum()

average_dosage = total_dosage/count
totally_clean_average_percentage = count_superior_dosage/count*100

print("Area livre total:", area, 'm^2')
print('Área totalmente limpa:', totally_clean_average_percentage, '%')
print('Dosagem média por área:', average_dosage, 'mJ/cm^2')
minutes = simulation.time_required//60
seconds = simulation.time_required-minutes*60
print('Tempo real:', minutes, 'min:', seconds, 'seg')
