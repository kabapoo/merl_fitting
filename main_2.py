import time
import numpy as np
import math as m

import merl
import coord
import nlls

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0

SAMPLE_RATE = 4
FILEPATH = "../MERLDatabase/materials/"
FILENAME = "blue-metallic-paint.binary"

t1 = time.time()
brdf = merl.read_brdf(FILEPATH + FILENAME)
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

p0 = np.array([0.5, 0.5, 0.5, 0.5])

pr = nlls.exe_least(p0, x, yr)
pg = nlls.exe_least(p0, x, yg)
pb = nlls.exe_least(p0, x, yb)
t4 = time.time()
print("nlls에 걸린 시간", t4 - t3)

print(pr.x)
print(pg.x)
print(pb.x)

fp = open("merl_params.txt", "a")
fp.write(FILENAME+"\n")
fp.write("channel,diffuse,specular,roughness,fresnel\n")
line1 = "red,{0:f},{1:f},{2:f},{3:f}\n".format(pr.x[0], pr.x[1], pr.x[2], pr.x[3])
line2 = "green,{0:f},{1:f},{2:f},{3:f}\n".format(pg.x[0], pg.x[1], pg.x[2], pg.x[3])
line3 = "blue,{0:f},{1:f},{2:f},{3:f}\n".format(pb.x[0], pb.x[1], pb.x[2], pb.x[3])
fp.write(line1+line2+line3)
fp.close()