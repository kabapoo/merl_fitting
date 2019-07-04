import merl
import render

import matplotlib.pyplot as plt
import math as m
import numpy as np

FILEPATH = "../MERLDatabase/materials/"
FILENAME = "green-metallic-paint.binary"

brdf = merl.read_brdf(FILEPATH + FILENAME)

theta_half = np.arange(0.0, 1.57, 0.01)
phi_half = np.repeat(0.0, 157)
theta_diff = np.repeat(0.0, 157)
phi_diff = np.repeat(0.0, 157)

param_red = [0.2482, 0.0008, 0.5529, 0.0000000057]
param_green = [0.33108, 0.000024, 0.001188, 0.000000113]
param_blue = [0.34824, 0.000022, 0.001053, 0.0000000651]
yr_render = render.cook_3d(theta_half, theta_diff, phi_diff, param_red)
yg_render = render.cook_3d(theta_half, theta_diff, phi_diff, param_green)
yb_render = render.cook_3d(theta_half, theta_diff, phi_diff, param_blue)

theta_in, phi_in, theta_out, phi_out = render.np_half_diff_to_in_out(theta_half, theta_diff, phi_diff)

yr = []
yg = []
yb = []
for i in range(theta_in.size):
    yr.append(merl.lookup_brdf_val(brdf, theta_in[i], phi_in[i], theta_out[i], phi_out[i])[0])
    yg.append(merl.lookup_brdf_val(brdf, theta_in[i], phi_in[i], theta_out[i], phi_out[i])[1])
    yb.append(merl.lookup_brdf_val(brdf, theta_in[i], phi_in[i], theta_out[i], phi_out[i])[2])

yr_merl = np.array(yr)
yg_merl = np.array(yg)
yb_merl = np.array(yb)

plt.plot(theta_half, yb_render, 'ro')
plt.plot(theta_half, yb_merl, 'bo')
plt.show()
