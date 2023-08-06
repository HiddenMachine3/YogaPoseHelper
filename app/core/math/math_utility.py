import numpy as np
import math


def find_xy_plane_angle(a, vertex, c):
    """
    Returns the angle made between a,vertex and c
    """
    x1, y1, _ = a
    x2, y2, _ = vertex
    x3, y3, _ = c
    angle = abs(
        math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    )

    return angle


def find_angle(a, vertex, c):
    v1, v2 = np.array(a - vertex), np.array(c - vertex)
    v1_mag, v2_mag = np.linalg.norm(v1), np.linalg.norm(v2)

    if v1_mag == 0 or v2_mag == 0:
        return 0

    return abs(math.degrees(np.arccos(np.dot(v1, v2) / (v1_mag * v2_mag))))


def clamp(val, lower, upper):
    return min(max(lower, val), upper)
