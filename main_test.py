#test

import time
import numpy as np
import math as m

import merl
import coord
import render
import nlls

SAMPLE_RATE = 10

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

            x.append([theta_half, theta_diff, phi_diff])

x = np.array(x)
pr = np.array([0.3, 0.7, 0.02, 0.5])
pg = np.array([0.8, 0.7, 0.02, 0.5])
pb = np.array([0.7, 0.6, 0.02, 0.3])

r, g, b = render.render_3d_random(pr, pg, pb, x)
yr = np.array(r)

p0 = np.array([0.5, 0.5, 0.5, 0.5])

param_r = nlls.exe_least(p0, x, yr)

print(param_r)
