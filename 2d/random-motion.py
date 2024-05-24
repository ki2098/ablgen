import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anima
import math

nstep = 100000

dt = 1e-4

mu, sigma = 0.0, 1.0
fxt1 = np.random.normal(mu, sigma, nstep)
fyt1 = np.random.normal(mu, sigma, nstep)
fxt2 = np.random.normal(mu, sigma, nstep)
fyt2 = np.random.normal(mu, sigma, nstep)

img = []

fig = plt.figure()
ax = fig.add_subplot(111)

x1 = np.zeros(nstep)
y1 = np.zeros(nstep)
x1[0] = 0
y1[0] = 0 
u1 = 0
v1 = 0


x2 = np.zeros(nstep)
y2 = np.zeros(nstep)
x2[0] = 0
y2[0] = 0 
u2 = 0
v2 = 0

def move(fx, fy, x, y):
    for i in range(1, nstep):
        xp = x[i-1]
        yp = y[i-1]
        r = math.sqrt(xp*xp + yp*yp)
        th = math.atan2(yp, xp)
        eff = 1+r
        dr = fx[i-1]*math.cos(th) + fy[i-1]*math.sin(th)
        dz = fx[i-1]*math.sin(th) - fy[i-1]*math.cos(th)
        if dr > 0:
            dr = dr / eff
        else:
            dr = dr * eff
        dx = dr*math.cos(th) + dz*math.sin(th)
        dy = dr*math.sin(th) - dz*math.cos(th)
        # print("%12.5e, %12.5e, %12.5e, %12.5e, %12.5e, %12.5e, %12.5e, %12.5e, %12.5e"%(xp, yp, r, fx[i-1], fy[i-1], dr, dz, dx, dy))
        x[i] = xp + dx*dt
        y[i] = yp + dy*dt

move(fxt1, fyt1, x1, y1)
move(fxt2, fyt2, x2, y2)

ax.plot(x1, y1, linewidth=0.5, color='b')
ax.plot(x2, y2, linewidth=0.5, color='r')
ax.set_aspect('equal')
ax.grid(True)
fig.savefig('random-motion.jpg')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x1, y1, s=0.1, c='b')
ax.scatter(x2, y2, s=0.1, c='r')
ax.set_aspect('equal')
ax.grid(True)
fig.savefig('random-motion-cloud.jpg')
