import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def np_angle_to_coords(theta, phi):
    _z = np.cos(theta)    
    _x = np.sin(theta) * np.cos(phi)
    _y = np.sin(theta) * np.sin(phi)
    _vec = np.array([_x, _y, _z]).transpose()
    return _vec

def np_coords_to_angle(vec):
    theta = np.arccos(vec[:,2])
    phi = np.arctan2(vec[:,1], vec[:,0])
    return (theta, phi)

def np_normalize(vec):
    for i in range(vec.shape[0]):
        norm = np.linalg.norm(vec[i])
        if norm != 0: 
            vec[i] = vec[i] / norm

def np_half_angle(theta_in, phi_in, theta_out, phi_out):
    in_vec = np_angle_to_coords(theta_in, phi_in)
    out_vec = np_angle_to_coords(theta_out, phi_out)
    half_vec = in_vec + out_vec
    np_normalize(half_vec)
    return np_coords_to_angle(half_vec)

def np_dot_product(v1, v2):
    if v1.shape[0] != v2.shape[0]:
        print("np_dot_product: row size error")
        return 0.0
    if v1.shape[1] != v2.shape[1]:
        print("np_dot_product: col size error")
        return 0.0

    result = np.zeros((v1.shape[0]))
    for i in range(v1.shape[1]):
        plus = (v1[:,i] * v2[:,i])
        result = result + plus
    return result

def np_row_multiply(vec, row):
    # (m, n) * (m, 1) => (m, n)
    if vec.shape[0] != row.size:
        print("np_row_multiply: dimension error ({0}, {1})".format(vec.shape[0], row.size))
        return np.zeros(vec.shape)
    return np.array([vec[:,i] * row for i in range(vec.shape[1])]).transpose()

def np_rotate_vector(vec, axis, angle):
    _cos = np.cos(angle)
    out = np_row_multiply(vec, _cos)
    I = np.ones((vec.shape[0], 1))
    axis_ex = axis * I
    temp = np.dot(vec, axis) * (1.0 - np.cos(angle))
    temp = np.expand_dims(temp, axis=1)
    out = out + (axis * temp)
    crs = np.cross(axis_ex, vec)
    _sin = np.sin(angle)
    out = out + np_row_multiply(crs, _sin)
    return out    

def np_angle_to_dots(theta_in, phi_in, theta_out, phi_out):
    ndotl = np.cos(theta_in)
    theta_half, phi_half = np_half_angle(theta_in, phi_in, theta_out, phi_out)
    ndoth = np.cos(theta_half)
    ndotv = np.cos(theta_out)
    v_vec = np_angle_to_coords(theta_out, phi_out)
    h_vec = np_angle_to_coords(theta_half, phi_half)
    vdoth = np_dot_product(v_vec, h_vec)
    return (ndotl, ndoth, ndotv, vdoth)

def np_half_diff_to_in_out(theta_half, theta_diff, phi_diff):
    bi_normal = np.array([0.0, 1.0, 0.0])
    diff = np_angle_to_coords(theta_diff, phi_diff)
    z_axis = np.array([0.0, 0.0, 1.0])
    half = np_angle_to_coords(theta_half, np.zeros(theta_half.size))
    in_vec = np_rotate_vector(diff, bi_normal, theta_half)
    proj_half = np_row_multiply(half, np_dot_product(in_vec, half))
    out_vec = proj_half * 2.0 - in_vec
    np_normalize(out_vec)
    theta_in, phi_in = np_coords_to_angle(in_vec)
    theta_out, phi_out = np_coords_to_angle(out_vec)
    return (theta_in, phi_in, theta_out, phi_out)

def cook_angle(theta_in, phi_in, theta_out, phi_out, p):
    ndotl, ndoth, ndotv, vdoth = np_angle_to_dots(theta_in, phi_in, theta_out, phi_out)

    diff = p[0] * ndotl
    
    fresnel = p[3] + ((1.0 - p[3]) * (1.0 - vdoth) ** 5.0)

    g1 = (2 * ndoth * ndotv) / vdoth
    g2 = (2 * ndoth * ndotl) / vdoth
    geometry = np.minimum(1.0, np.minimum(g1, g2))

    r1 = 1.0 / (math.pi * p[2] ** 2 * ndoth ** 4)
    r2 = (ndoth ** 2 - 1.0) / (p[2] ** 2 * ndoth ** 2)
    distribution = r1 * np.exp(r2)

    spec = p[1] * (fresnel * geometry * distribution) / (math.pi * ndotv * ndotl)

    return diff + spec

def cook_3d(theta_half, theta_diff, phi_diff, p):
    theta_in, phi_in, theta_out, phi_out = np_half_diff_to_in_out(theta_half, theta_diff, phi_diff)
    ndotl, ndoth, ndotv, vdoth = np_angle_to_dots(theta_in, phi_in, theta_out, phi_out)
        
    diff = p[0] * ndotl
    
    fresnel = p[3] + ((1.0 - p[3]) * (1.0 - vdoth) ** 5.0)

    g1 = (2 * ndoth * ndotv) / vdoth
    g2 = (2 * ndoth * ndotl) / vdoth
    geometry = np.minimum(1.0, np.minimum(g1, g2))

    r1 = 1.0 / (math.pi * p[2] ** 2 * ndoth ** 4)
    r2 = (ndoth ** 2 - 1.0) / (p[2] ** 2 * ndoth ** 2)
    distribution = r1 * np.exp(r2)

    spec = p[1] * (fresnel * geometry * distribution) / (math.pi * ndotv * ndotl)

    return diff + spec

def cook_3d_random(theta_half, theta_diff, phi_diff, p):
    result = cook_3d(theta_half, theta_diff, phi_diff, p)
    result = result + np.random.normal(0.0, 0.1, result.size)
    return result

def draw3d(p):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    theta_half = np.arange(0.0, 1.57, 0.01)
    phi_diff = np.arange(0.0, 1.57, 0.01)
    theta_diff = np.repeat(0.0, 157)
    phi_half = np.repeat(0.0, 157)

    theta_half, phi_diff = np.meshgrid(theta_half, phi_diff)
    
    y = cook_3d(theta_half, theta_diff, phi_diff, p)
    
    surf = ax.plot_surface(theta_half, phi_diff, y, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(0.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def render_all(sample_size, pr, pg, pb):
    r = []
    g = []
    b = []
    for i in range(sample_size):
        theta_in = i * 0.5 * math.pi / sample_size
        for j in range(sample_size * 4):
            phi_in = j * 2.0 * math.pi / (sample_size * 4)
            for k in range(sample_size):
                theta_out = k * 0.5 * math.pi / sample_size
                for l in range(sample_size * 4):
                    phi_out = l * 2.0 * math.pi / (sample_size * 4)
                    r.append(cook_angle(theta_in, phi_in, theta_out, phi_out, pr))
                    g.append(cook_angle(theta_in, phi_in, theta_out, phi_out, pg))
                    b.append(cook_angle(theta_in, phi_in, theta_out, phi_out, pb))
    return (r, g, b)

def render_3d(pr, pg, pb, x):
    r = cook_3d(x[:,0], x[:,1], x[:,2], pr)
    g = cook_3d(x[:,0], x[:,1], x[:,2], pg)
    b = cook_3d(x[:,0], x[:,1], x[:,2], pb)
            
    return (r, g, b)

def render_3d_random(pr, pg, pb, x):
    r = cook_3d_random(x[:,0], x[:,1], x[:,2], pr)
    g = cook_3d_random(x[:,0], x[:,1], x[:,2], pg)
    b = cook_3d_random(x[:,0], x[:,1], x[:,2], pb)
            
    return (r, g, b)

def rmse(y1, y2):
    result = 0.0
    if y1.size != len(y2):
        print("rmse: size error")
        return result
    for i in range(len(y2)):
        result += (y1[i] - y2[i]) ** 2
    result /= len(y2)
    return result

if __name__ == "__main__":
    p = np.array([0.5, 0.5, 0.05, 1.0])
    theta_half = np.arange(0.0, 1.57, 0.01)
    phi_half = np.repeat(0.0, 157)
    theta_diff = np.repeat(0.0, 157)
    phi_diff = np.repeat(0.0, 157)

    y = cook_3d_random(theta_half, theta_diff, phi_diff, p)
    plt.plot(theta_half, y)
    plt.show()
