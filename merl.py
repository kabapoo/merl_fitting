import time
import math as m
import struct
import numpy as np
import matplotlib.pyplot as plt

import coord

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0

def theta_half_index(theta_half):
    if theta_half <= 0.0:
        return 0
    theta_half_deg = ((theta_half / (m.pi / 2.0)) * BRDF_SAMPLING_RES_THETA_H)
    temp = theta_half_deg * BRDF_SAMPLING_RES_THETA_H
    temp = m.sqrt(temp)
    ret_val = int(temp)
    if ret_val < 0:
        ret_val = 0
    if ret_val >= BRDF_SAMPLING_RES_THETA_H:
        ret_val = BRDF_SAMPLING_RES_THETA_H - 1
    return ret_val

def theta_diff_index(theta_diff):
    tmp = int(theta_diff / (m.pi * 0.5) * BRDF_SAMPLING_RES_THETA_D)
    if tmp < 0:
        return 0
    elif tmp < BRDF_SAMPLING_RES_THETA_D - 1:
        return tmp
    else:
        return BRDF_SAMPLING_RES_THETA_D - 1

def phi_diff_index(phi_diff):
    if phi_diff < 0.0:
        phi_diff += math.pi
    
    tmp = int(phi_diff / m.pi * BRDF_SAMPLING_RES_PHI_D / 2)
    if tmp < 0:
        return 0
    elif tmp < int(BRDF_SAMPLING_RES_PHI_D / 2) - 1:
        return tmp
    else:
        return int(BRDF_SAMPLING_RES_PHI_D / 2) - 1

def lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out):
    theta_half, phi_half, theta_diff, phi_diff = coord.std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out)
    
    ind = phi_diff_index(phi_diff) + theta_diff_index(theta_diff) * int(BRDF_SAMPLING_RES_PHI_D / 2) + theta_half_index(theta_half) * int(BRDF_SAMPLING_RES_PHI_D / 2) * BRDF_SAMPLING_RES_THETA_D
    red_val = brdf[ind] * RED_SCALE
    green_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * int(BRDF_SAMPLING_RES_PHI_D / 2)] * GREEN_SCALE
    blue_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D] * BLUE_SCALE

    if red_val < 0.0 or green_val < 0.0 or blue_val < 0.0:
        #print("Below horizon")
        red_val = 0.0
        green_val = 0.0
        blue_val = 0.0
    return (red_val, green_val, blue_val)

def read_brdf(filename):
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype='int', count=3)
        n = dim[0] * dim[1] * dim[2]
        brdf = np.fromfile(f, dtype='float', count=3 * n)
    return brdf

if __name__ == "__main__":
    import render

    FILEPATH = "../MERLDatabase/materials/"
    FILENAME = "green-metallic-paint.binary"
    SAMPLE_RATE = 10

    t1 = time.time()
    brdf = read_brdf(FILEPATH + FILENAME)
    t2 = time.time()
    print("merl 로딩 시간", t2 - t1)

    x = []
    r = []
    g = []
    b = []
    for ind_theta_half in range(0,90,SAMPLE_RATE):
        for ind_theta_diff in range(0,90,SAMPLE_RATE):
            for ind_phi_diff in range(0,180,SAMPLE_RATE):
                ind = ind_phi_diff + ind_theta_diff * 180 + ind_theta_half * 180 * 90
                theta_half = ind_theta_half ** 2 * 0.5 * m.pi / 90 ** 2
                ind_diff = ind % 16200
                theta_diff = (ind_diff // 180) * 0.5 * m.pi / 90
                phi_diff = (ind_diff % 180) * m.pi / 180

                red_val = brdf[ind] * RED_SCALE
                green_val = brdf[ind + 90 * 90 * 180] * GREEN_SCALE
                blue_val = brdf[ind + 90 * 90 * 360] * BLUE_SCALE

                if red_val > 0.0 and green_val > 0.0 and blue_val > 0.0:
                    x.append([theta_half, theta_diff, phi_diff])
                    r.append(red_val)
                    g.append(green_val)
                    b.append(blue_val)

    x = np.array(x)
    yr = np.array(r)
    yg = np.array(g)
    yb = np.array(b) 
    t3 = time.time()
    print("변환에 걸린 시간", t3 - t2)
    print("vector size:", x.shape)
    print("value size:", yr.shape)

    theta_in, phi_in, theta_out, phi_out = render.np_half_diff_to_in_out(x[:,0], x[:,1], x[:,2])
    ndotl, ndoth, ndotv, vdoth = render.np_angle_to_dots(theta_in, phi_in, theta_out, phi_out)

    p = np.array([0.344295, 0.562, 0.020548, 1.0])
    rr = render.cook_3d(x[:,0], x[:,1], x[:,2], p)

    plt.plot(ndoth, yr, 'ro', markersize=1)
    #plt.plot(ndoth, rr, 'go', markersize=1)
    plt.show()
