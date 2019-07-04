
# 모든 theta_in, phi_in, theta_out, phi_out <-> theta_half, theta_diff, phi_diff test

# vdoth가 왜 -가 나오는가?

# theta_in, phi_in 고정하고 theta_out, phi_out을 축으로 3d plot

import render
import merl

import numpy as np
import math as m

sample = 15
x = []
xx = []
for i in range(sample):
    theta_in = i * m.pi * 0.5 / sample
    for j in range(sample):
        phi_in = j * m.pi * 1.0 / sample
        for k in range(sample):
            theta_out = k * m.pi * 0.5 / sample
            for l in range(sample):
                phi_out = l * m.pi * 1.0 / sample
                x.append([theta_in, phi_in, theta_out, phi_out])
                theta_half, _, theta_diff, phi_diff = merl.coord.std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out)
                xx.append([theta_half, theta_diff, phi_diff])

std_angle = np.array(x)
three_angle = np.array(xx)

a, b, c, d = render.np_half_diff_to_in_out(three_angle[:,0], three_angle[:,1], three_angle[:,2])
'''
f = open("angle_test.txt", "w")

for i in range(std_angle.shape[0]):
    line1 = "{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f} -> ".format(std_angle[i,0], std_angle[i,1], std_angle[i,2], std_angle[i,3])
    line2 = "{0:.2f}, {1:.2f}, {2:.2f} -> ".format(three_angle[i,0], three_angle[i,1], three_angle[i,2])
    line3 = "{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}\n".format(a[i], b[i], c[i], d[i])
    f.write(line1 + line2 + line3)

f.close()
'''
ndotl, ndoth, ndotv, vdoth = render.np_angle_to_dots(a, b, c, d)
theta_half, phi_half = render.np_half_angle(std_angle[:,0], std_angle[:,1], std_angle[:,2], std_angle[:,3])
_in = render.np_angle_to_coords(std_angle[:,0], std_angle[:,1])
_out = render.np_angle_to_coords(std_angle[:,2], std_angle[:,3])
_half = render.np_angle_to_coords(theta_half, phi_half)
ldoth = render.np_dot_product(_in, _half)

for i in range(vdoth.shape[0]):
    if _in[i][0] < 0.0 and _out[i][0] > 0.0 and _out[i][2] - _in[i][2] < 0.2 and vdoth[i] < 0.2:
        print()
        print("{0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5f}, {5:.5f}\t".format(std_angle[i][0], std_angle[i][1], std_angle[i][2], std_angle[i][3], theta_half[i], phi_half[i]), end='')
        print("{0:.5f}, {1:.5f}, {2:.5f}\t".format(three_angle[i][0], three_angle[i][1], three_angle[i][2]), end='')
        print("{0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}".format(a[i], b[i], c[i], d[i]))
        print(_in[i], _out[i], _half[i], ldoth[i], vdoth[i])

# l.h < 0 이라는 것은
# z축 . diff_vec가 < 0 이었을 수도 있다.
# rotate 방향 테스트
# half vector 이상함
# theta를 구하는데 normalize 안된 z값으로 하면 에러
