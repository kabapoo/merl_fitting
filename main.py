import time
import numpy as np

#import merl
import coord
import import_csv
import render
import nlls

SAMPLESIZE = 16

#brdf = merl.read_brdf('../../MERLDatabase/materials/green-metallic-paint.binary')
light = np.array([1.0, 1.0, 1.0])
light = coord.normalize(light)
view = np.array([0.0, 1.0, 0.0])
view = coord.normalize(view)

t1 = time.time()
#x, yr, yg, yb = nlls.load_all(brdf, SAMPLESIZE)
#x, yr, yg, yb = nlls.load_sub(brdf, SAMPLESIZE, light, view)
x = []
y = []
import_csv.importCSV('../../MERLDatabase/CSV/aluminium.csv', x, y)
t2 = time.time()
x = np.array(x)
y = np.array(y)
yr = y[:,0]
yg = y[:,1]
yb = y[:,2]
t3 = time.time()
print("로딩에 걸린 시간", t2 - t1)
print("변환에 걸린 시간", t3 - t2)
print("vector size:", x.shape)
print("rgb size:", y.shape)

p0 = np.array([0.5, 0.5, 0.5, 0.5])

pr = nlls.exe_least(p0, x, yr)
pg = nlls.exe_least(p0, x, yg)
pb = nlls.exe_least(p0, x, yb)
t4 = time.time()
print("nlls에 걸린 시간", t4 - t3)

print(pr.x)
print(pg.x)
print(pb.x)

r, g, b = render.render(pr.x, pg.x, pb.x, x)
render.draw3d(pg.x)

rmse = 0.0
if yr.size == len(r):
    for i in range(len(r)):
        rmse += (yr[i] - r[i]) ** 2
    rmse /= len(r)
print(rmse) # red
rmse = 0.0
if yg.size == len(g):
    for i in range(len(g)):
        rmse += (yg[i] - g[i]) ** 2
    rmse /= len(g)
print(rmse) # green
rmse = 0.0
if yb.size == len(b):
    for i in range(len(b)):
        rmse += (yb[i] - b[i]) ** 2
    rmse /= len(b)
print(rmse) # green