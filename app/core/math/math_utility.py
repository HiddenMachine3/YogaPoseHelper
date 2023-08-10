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

def find_3d_angle(p1, vertex, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    xv, yv, zv = vertex
    dx1 = x1 - xv
    dy1 = y1 - yv
    dz1 = z1 - zv
    dx2 = x2 - xv
    dy2 = y2 - yv
    dz2 = z2 - zv
    dot_product = dx1 * dx2 + dy1 * dy2 + dz1 * dz2
    mag1 = math.sqrt(dx1 ** 2 + dy1 ** 2 + dz1 ** 2)
    mag2 = math.sqrt(dx2 ** 2 + dy2 ** 2 + dz2 ** 2)
    angle = math.acos(dot_product / (mag1 * mag2))
    return math.degrees(angle)

def find_angle(a, vertex, c):
    v1, v2 = np.array(a - vertex), np.array(c - vertex)
    v1_mag, v2_mag = np.linalg.norm(v1), np.linalg.norm(v2)

    if v1_mag == 0 or v2_mag == 0:
        return 0

    return abs(math.degrees(np.arccos(np.dot(v1, v2) / (v1_mag * v2_mag))))


def clamp(val, lower, upper):
    return min(max(lower, val), upper)
