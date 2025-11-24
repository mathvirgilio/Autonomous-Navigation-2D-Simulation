import pygame, sys
import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
global WINDOW_HEIGHT
global WINDOW_WIDTH

dose_necessary = 16.9 #mJ/cm^2
power = 60 #W
attenuation = 10 #%
exposure_time = 60 #seg

def initial_drawGrid(SCREEN, navegation_map):
    walls = []
    blocks_surf = []
    navegation_map = navegation_map.T
    blockSize = 25 #Set the size of the grid block
    shape_x = navegation_map.shape[0]
    shape_y = navegation_map.shape[1]
    for i in range(shape_x):
        for j in range(shape_y):
            x = i*blockSize
            y = j*blockSize
            rect = pygame.Rect(x, y, blockSize, blockSize)
            if (navegation_map[i][j] == '0'):
                pygame.draw.rect(SCREEN, (255,0,255),rect, 0)
            if (navegation_map[i][j] == '1'):
                pygame.draw.rect(SCREEN, BLACK,rect, 0)
                walls.append((x,y))
                blocks_surf.append(rect)
                
    return walls, blocks_surf

def uv_irradiate(walls, robotrect, rect):
    x_robot, y_robot = robotrect.center[0], robotrect.center[1]
    x_block, y_block = rect.center[0], rect.center[1]
    k = 5
    
    if(x_robot == x_block):
        #print(1)
        x = x_robot
        aux = (y_block-y_robot)//abs(y_block-y_robot)
        for y in range(y_robot, y_block, k*aux):
             for wall in walls:
                 do_not_irradiate = (x > wall[0]) and (x < (wall[0] + 25)) and (y > wall[1]) and (y < (wall[1] + 25))
                 if(do_not_irradiate):

                     return 0
        return 1
    
    else:
        aux = (x_block-x_robot)//abs(x_block-x_robot)
        a = (y_block-y_robot)/(x_block-x_robot)
        
        av = (x_robot + x_block)//2
        
        for i in range(0, x_block-av, k*aux):
            x1 = av+i
            y1 = y_robot + a*(x1-x_robot)
            x2 = av-i
            y2 = y_robot + a*(x2-x_robot)
             
            for wall in walls:
                do_not_irradiate = (x1 > wall[0]) and (x1 < (wall[0] + 25)) and (y1 > wall[1]) and (y1 < (wall[1] + 25))
                if(do_not_irradiate):
                    return 0
                do_not_irradiate = (x2 > wall[0]) and (x2 < (wall[0] + 25)) and (y2 > wall[1]) and (y2 < (wall[1] + 25))
                if(do_not_irradiate):
                    return 0

        return 1


                         