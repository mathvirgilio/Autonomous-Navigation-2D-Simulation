import math
import numpy as np
import sys
import simulation as obj
import pygame
import matplotlib.pyplot as plt


#Informações UV-C
necessary_dosage = 16.9 #mJ/cm^2
power = 60 #W
attenuation = 10 #%
exposure_time = 0 #segundos
time_delay = 1 #milisegundos

file_name = 'mapa_britto.txt' #Arquivo com o mapa
initial_pos = np.array([4, 8]) #Ponto de partida
robot_dim = 2 #Dimensões do robô
sensor_range = 4

def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1] #mudar sinal
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

def read_scanning(distances_per_angle, raster_shape=(200,200), scale=1, depth = 2, delta=np.zeros(3)):
    print("Começou")
    raster = np.zeros(shape=raster_shape)
    i_center, j_center = (raster_shape[0])/2, (raster_shape[1])/2
    for angle, distance in enumerate(distances_per_angle):
        if(distance <= 95):
            radians = (angle)*np.pi/180
            if(scale == 1):
                displaciment = distance*np.math.sin(radians)
                i = i_center + displaciment
                displaciment = distance*np.math.cos(radians)
                j = j_center + displaciment
                if(np.any(delta)):
                    x, y, theta = delta
                    (i, j) = rotatePoint([i_center + x, j_center+y], [i + x, j + y], theta)
                
                i, j = round(i), round(j)
            else:
                displaciment = (distance+scale)*np.math.sin(radians)//scale
                i = i_center + displaciment
                displaciment = (distance+scale)*np.math.cos(radians)//scale
                j = j_center + displaciment
                i, j = round(i), round(j)
            print("Angulo:", angle, "D:", distance,  "(i, j):", i, j)

            for di in range(-depth, depth + 1):
                for dj in range(-depth, depth + 1):
                    if((i + di) >= 0 and (i + di) < raster_shape[0]):
                        if((j + dj) >= 0 and (j + dj) < raster_shape[1]):
                            raster[i+di][j+dj] = 1
        
    return raster
   
#Simulador
if __name__ == '__main__':
    simulation = obj.simulation()
    simulation.read_map(file_name)
    simulation.create_robot(initial_pos, robot_dim)
    simulation.create_display()

    simulation.show_navegation(power, necessary_dosage, attenuation, scan_on=True)
    raster = read_scanning(simulation.distances)
    raster2 = read_scanning(simulation.distances, raster_shape=(6,6), scale=25, depth = 1)
    #simulation.show_navegation(power, necessary_dosage, attenuation, scan_on=True, delta = (5, 5, 30))
    #raster2 = read_scanning(simulation.distances, delta = (50, 50, 30))
    #raster2 = read_scanning(simulation.distances)
    
    raster = np.transpose(raster)
    #raster = np.rot90(raster)
    raster2 = np.transpose(raster2)
    #raster2 = np.rot90(raster2)
    print(raster2)
    upway_available = not np.any(raster2[1, 2:4])
    leftway_available = not np.any(raster2[2:4, 1])
    rightway_available = not np.any(raster2[2:4, 4])
    downway_available = not np.any(raster2[4, 2:4])
    print(np.array([upway_available, leftway_available, rightway_available, downway_available]))

    plt.imshow(raster,cmap='gray_r')
    plt.show()
    plt.imshow(raster2,cmap='gray_r')
    plt.show()
    pygame.quit()
    sys.exit()