import math
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

import merl
import render

def obj_func(p, x, y):
    theta_half = x[:,0]
    theta_diff = x[:,1]
    phi_diff = x[:,2]
    return y - render.cook_3d(theta_half, theta_diff, phi_diff, p)

def load_all(brdf, sample_size):
    x = []
    yr = []
    yg = []
    yb = []
    for i in range(sample_size):
        theta_in = i * 0.5 * math.pi / sample_size
        for j in range(sample_size * 4):
            phi_in = j * 2.0 * math.pi / (sample_size * 4)
            for k in range(sample_size):
                theta_out = k * 0.5 * math.pi / sample_size
                for l in range(sample_size * 4):
                    phi_out = l * 2.0 * math.pi / (sample_size * 4)
                    r, g, b = merl.lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out)
                    x.append([theta_in, phi_in, theta_out, phi_out])
                    yr.append(r)
                    yg.append(g)
                    yb.append(b)
    return (x, yr, yg, yb)

def load_sub(brdf, sample_size, light, view):
    x = []
    yr = []
    yg = []
    yb = []
    for i in range(sample_size):
        theta = (math.pi * i / sample_size)
        for j in range(sample_size):
            phi = math.pi * j / sample_size
            normal = merl.coord.angle_to_coords(theta, phi)
            z_axis = np.array([0.0, 0.0, 1.0])
            theta_z = math.acos(np.dot(normal, z_axis))
            axis = np.cross(normal, z_axis)
            axis = merl.coord.normalize(axis)
            light = merl.coord.rotate_vector(light, axis, theta_z)
            view = merl.coord.rotate_vector(view, axis, theta_z)
            theta_in, phi_in = merl.coord.coords_to_angle(light)
            theta_out, phi_out = merl.coord.coords_to_angle(view)
            r, g, b = merl.lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out)
            x.append([theta_in, phi_in, theta_out, phi_out])
            yr.append(r)
            yg.append(g)
            yb.append(b)
    return (x, yr, yg, yb)

def exe_least(p0, x, y):
    p = least_squares(obj_func, p0, method='trf', bounds=(0.0, np.inf), args=(x, y), verbose=0)
    return p

if __name__ == "__main__":    
    brdf = merl.read_brdf('green-metallic-paint.binary')
    x, yr, yg, yb = load_all(brdf, 4)
    light = np.array([1.0, 1.0, 1.0])
    light = merl.coord.normalize(light)
    view = np.array([0.0, 1.0, 0.0])
    view = merl.coord.normalize(view)
    #x, yr, yg, yb = load_sub(brdf, 8, light, view)
    x = np.array(x)
    yr = np.array(yr)
    yg = np.array(yg)
    yb = np.array(yb)

    p0_r = np.array([0.5, 0.5, 0.5, 0.5])
    p_r = least_squares(obj_func, p0_r, method='trf', bounds=(0.0, np.inf), args=(x, yr), verbose=1)
    print(p_r)