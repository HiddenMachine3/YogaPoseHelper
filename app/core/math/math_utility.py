import numpy as np
import math

def findAngle(p1:np.array, p2:np.array, p3:np.array):
    v1 = p1-p2
    v2 = p3-p2
    v1_mag,v2_mag = np.linalg.norm(np.array([v1[0],v1[1]])), np.linalg.norm(np.array([v2[0],v2[1]]))
    
    if(v1_mag == 0 or v2_mag == 0): 
        print("trying to divide by zero prevented")
        return 0

    v1 = v1 / v1_mag; v2 = v2 / v2_mag
    # try:
    ans = math.acos(v1[0] * v2[0] + v1[1] * v2[1] ) #+ v1[2] * v2[2])
    # except ValueError as e:
    #     print(e)
    #     print(f"points:{p1},{p2},{p3}\nv1:{v1},v2:{v2}")
    return ans #math.degrees()  # v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def clamp(val, lower,upper):
    return min(max(lower,val),upper)