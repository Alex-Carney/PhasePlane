import scipy.integrate as si
import numpy as np
from numpy import ndarray
import datetime

r = 10

eqn1 = lambda t, x, y, z: 10*(-x+y)
eqn2 = lambda t, x, y, z: r*x - y - x*z
eqn3 = lambda t, x, y, z: -(8/3)*z + x*y

type(eqn1)

eqnSys = [eqn1, eqn2, eqn3]

t_start = 0
t_end = 1000

t_span = np.array([t_start, t_end])

ics = np.array([.2, .2, .1])

global eqnsystem
eqnsystem = [eqn1, eqn2, eqn3]

def ode_sys(t, XYZ):
    dxdt = eqnsystem[0](t, XYZ[0], XYZ[1], XYZ[2])
    dydt = eqnsystem[1](t, XYZ[0], XYZ[1], XYZ[2])
    dzdt = eqnsystem[2](t, XYZ[0], XYZ[1], XYZ[2])
    return [dxdt, dydt, dzdt]

begin_time = datetime.datetime.now()
out = si.solve_ivp(ode_sys, np.array([0, 100]), ics, method='LSODA')
print("elapsed: " + str(datetime.datetime.now() - begin_time))

xout = out.y[0, :]
yout = out.y[1, :]
zout = out.y[2, :]

print(max(xout))
print(max(yout))
print(max(zout))






