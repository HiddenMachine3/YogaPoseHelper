import numpy as np
import math

def find_xy_plane_angle(a,vertex,c):
    """
    Returns the angle made between a,vertex and c
    """
    x1,y1,_=a
    x2,y2,_=vertex
    x3,y3,_=c
    angle=abs(math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2)))

    return angle

def clamp(val, lower,upper):
    return min(max(lower,val),upper)