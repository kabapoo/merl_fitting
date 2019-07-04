import time
import numpy as np
import math as m

import merl
import coord
import render

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0

SAMPLESIZE = 16
FILEPATH = "../MERLDatabase/materials/"
FILENAME = "aluminium.binary"

t1 = time.time()
brdf = merl.read_brdf(FILEPATH + FILENAME)
t2 = time.time()
print("merl 로딩 시간", t2 - t1)

x = []
r = []
g = []
b = []
for ind_theta_half in range(90):
    for ind_theta_diff in range(90):
        for ind_phi_diff in range(180):
            ind = ind_phi_diff + ind_theta_diff * 180 + ind_theta_half * 180 * 90
            theta_half = ind_theta_half ** 2 * 0.5 * m.pi / 90 ** 2
            ind_diff = ind % 16200
            theta_diff = (ind_diff // 180) * 0.5 * m.pi / 90
            phi_diff = (ind_diff % 180) * m.pi / 180

            x.append([theta_half, theta_diff, phi_diff])
            r.append(brdf[ind] * RED_SCALE)
            g.append(brdf[ind + 90 * 90 * 180] * GREEN_SCALE)
            b.append(brdf[ind + 90 * 90 * 360] * BLUE_SCALE)

x = np.array(x)
yr = np.array(r)
yg = np.array(g)
yb = np.array(b) 
t3 = time.time()
print("변환에 걸린 시간", t3 - t2)
print("vector size:", x.shape)
print("value size:", yr.shape)

param_red = [1.000000,0.000179,0.002622,1.000000]
param_green = [1.000000,0.000217,0.002671,1.000000]
param_blue = [1.000000,0.000236,0.002982,1.000000]

r, g, b = render.render_3d(param_red, param_green, param_blue, x)

red_error = render.rmse(yr, r)
print("red_rmse:", red_error)
green_error = render.rmse(yg, g)
print("green_rmse:", green_error)
blue_error = render.rmse(yb, b)
print("blue_rmse:", blue_error)

fp = open("merl_rmse.txt", "a")
fp.write(FILENAME+"\n")
line = "red : {0}\t".format(red_error)
line = line + "green : {0}\t".format(green_error)
line = line + "blue : {0}]\n".format(blue_error)
fp.write(line)
fp.write("\n")
fp.close()
