import util.carney_diff_eqs as cde
import numpy as np
import matplotlib.pyplot as plt



dydx = lambda x, y: x+y

x_segments: int = 20
y_segments: int = 20
ymin: int = -3
ymax: int = 3
xmin: int = -4
xmax: int = 4

xpartition = np.linspace(xmin, xmax, x_segments)
ypartition = np.linspace(ymin, ymax, y_segments)

X, Y = np.meshgrid(xpartition, ypartition)

dy = dydx(X, Y)
dx = np.ones(dy.shape)

M = np.sqrt(dx**2 + dy**2) #magnitude

xInput = -4
yInput = 3


x_step: float = .1
x_domainRight = np.r_[xInput:xmax*1.2:x_step]
x_domainLeft = np.r_[xInput:xmin*1.2:-x_step]
h = .1


outRight = cde.runge_kutta(dydx, x_domainRight, yInput, h)
outLeft = cde.runge_kutta(dydx, x_domainLeft, yInput, -h)

plt.quiver(xpartition, ypartition, dx/M, dy/M, headaxislength=0, headlength=0)


x_domain = np.hstack((np.flip(x_domainLeft[1:]), x_domainRight))
out = np.hstack((np.flip(outLeft[1:]), outRight))


# plt.plot(x_domainRight, outRight)
# plt.plot(x_domainLeft, outLeft)

plt.plot(x_domain, out)


plt.ylim([ymin, ymax])
plt.xlim([xmin, xmax])
plt.hlines(0, xmin, xmax, 'k')
plt.vlines(0, ymin, ymax, 'k')
plt.show()



