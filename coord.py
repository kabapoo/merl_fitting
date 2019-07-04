import math
import numpy as np

def angle_to_coords(theta, phi):
    _z = math.cos(theta)    
    _x = math.sin(theta) * math.cos(phi)
    _y = math.sin(theta) * math.sin(phi)
    _vec = np.array([_x, _y, _z])
    return _vec

def coords_to_angle(vec):
    #theta = math.acos(min(max(vec[2], 0.0), 1.0))
    diag = math.sqrt(vec[2]**2 + vec[1]**2 + vec[0]**2)
    theta = math.acos(min(max((vec[2] / diag), 0.0), 1.0))
    phi = math.atan2(vec[1], vec[0])
    return (theta, phi)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def rotate_vector(vec, axis, angle):
    out = vec * math.cos(angle)
    temp = np.dot(axis, vec)
    temp = temp * (1.0 - math.cos(angle))
    out = out + (axis * temp)
    crs = np.cross(axis, vec)
    out = out + (crs * math.sin(angle))
    return out

def std_coords_to_half_coords(theta_in, phi_in, theta_out, phi_out):
    in_vec = angle_to_coords(theta_in, phi_in)
    in_vec = normalize(in_vec)

    out_vec = angle_to_coords(theta_out, phi_out)
    out_vec = normalize(out_vec)

    half_vec = in_vec + out_vec
    half_vec = normalize(half_vec)

    theta_half, phi_half = coords_to_angle(half_vec)
    return (theta_half, phi_half)

def std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out):
    in_vec = angle_to_coords(theta_in, phi_in)
    in_vec = normalize(in_vec)

    out_vec = angle_to_coords(theta_out, phi_out)
    out_vec = normalize(out_vec)

    half_vec = (in_vec + out_vec)
    half_vec = normalize(half_vec)

    theta_half, phi_half = coords_to_angle(half_vec)
    
    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    temp = rotate_vector(in_vec, normal, -phi_half)
    diff = rotate_vector(temp, bi_normal, -theta_half)

    theta_diff, phi_diff = coords_to_angle(diff)
    if phi_diff < 0.0:
        phi_diff += math.pi * 2.0
    return (theta_half, phi_half, theta_diff, phi_diff)

if __name__ == '__main__':
    theta_in = 1.46608
    phi_in = 2.93215
    theta_out = 1.25664
    phi_out = 0.0000
    theta_half, phi_half, theta_diff, phi_diff = std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out)
    print(theta_half, phi_half, theta_diff, phi_diff)

