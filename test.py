import time
import numpy as np
import math as m

import merl
import coord

SAMPLESIZE = 16

brdf = merl.read_brdf('../../MERLDatabase/materials/green-metallic-paint.binary')

light = np.array([1.0, 1.0, 1.0])
light = coord.normalize(light)
view = np.array([0.0, 1.0, 0.0])
view = coord.normalize(view)

theta_in = 1.23
phi_in = 0.78
theta_out = 0.78
phi_out = 0.0
print(theta_in, phi_in, theta_out, phi_out)

theta_half, phi_half, theta_diff, phi_diff = coord.std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out)
print(theta_half, phi_half, theta_diff, phi_diff)
ind_phi_diff= merl.phi_diff_index(phi_diff)
ind_theta_diff = merl.theta_diff_index(theta_diff)
ind_theta_half = merl.theta_half_index(theta_half)
print(ind_theta_half, ind_theta_diff, ind_phi_diff)

ind = merl.phi_diff_index(phi_diff) + merl.theta_diff_index(theta_diff) * 180 + merl.theta_half_index(theta_half) * 16200
print(ind)

r = brdf[ind] * merl.RED_SCALE
g = brdf[ind + 90 * 90 * 180] * merl.GREEN_SCALE
b = brdf[ind + 90 * 90 * 360] * merl.BLUE_SCALE

ind_theta_half = ind // 16200
ind_diff = ind % 16200
ind_theta_diff = ind_diff // 180
ind_phi_diff = ind_diff % 180

print(ind_theta_half, ind_theta_diff, ind_phi_diff)

theta_half = ind_theta_half ** 2 * 0.5 * m.pi / 8100
theta_diff = ind_theta_diff * 0.5 * m.pi / 90
phi_diff = ind_phi_diff * m.pi / 180
print(theta_half, theta_diff, phi_diff)

ind_phi_diff= merl.phi_diff_index(phi_diff)
ind_theta_diff = merl.theta_diff_index(theta_diff)
ind_theta_half = merl.theta_half_index(theta_half)
print(ind_theta_half, ind_theta_diff, ind_phi_diff)
