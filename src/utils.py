import math
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd

def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1] #mudar sinal
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

def relative_error(real, measured, bounds, length = 3):
    min_bound = np.ones(length)*(bounds[0] - 1) 
    max_bound = np.ones(length)*(bounds[1] - bounds[0] + 1) 
    norm_real = real - min_bound
    norm_real =  norm_real / max_bound
    norm_measured = measured - min_bound
    norm_measured = norm_measured / max_bound

    relative_error = abs(norm_real - norm_measured)/norm_real
    return relative_error