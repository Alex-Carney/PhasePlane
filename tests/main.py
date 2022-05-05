import numpy as np
import matplotlib.pyplot as plt
from util import carney_diff_eqs as ode44
import scipy.integrate as si
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# def main() -> None:
eqn1 = lambda x, y, t: -x+y
eqn2 = lambda x, y, t: y

time_step_size: float = .1
h = .1

tmax: int = 5
time = np.r_[0:10:time_step_size]  # type: ignore


x_segments: int = 20
y_segments: int = 20
ymin: int = -3
ymax: int = 3
xmin: int = -4
xmax: int = 4

xpartition = np.linspace(xmin, xmax, x_segments)
ypartition = np.linspace(ymin, ymax, y_segments)

X, Y = np.meshgrid(xpartition, ypartition)
U = eqn1(X, Y, time)
V = eqn2(X, Y, time)

M = np.sqrt(U**2 + V**2) #magnitude

plt.quiver(xpartition, ypartition, U / M, V / M, M, cmap = plt.cm.jet)

ics = np.array([1.5, .2])

outRight = ode44.runge_kutta_second(
    np.array([eqn1, eqn2]),
    np.r_[10:0:-time_step_size],
    ics,
    h)

outLeft = ode44.runge_kutta_second(
    np.array([eqn1, eqn2]),
    time,
    ics,
    -h)

out = np.hstack((np.flip(outLeft[:, 1:-1], axis=1)[:, 1:-1], (outRight[:, 1:-1])))[:, 1:-1]

# global eqnsystem
# eqnsystem = [eqn1, eqn2]
#
# def ode_sys(t, XYZ):
#     dxdt = eqnsystem[0](XYZ[0], XYZ[1], t)
#     dydt = eqnsystem[1](XYZ[0], XYZ[1], t)
#     return [dxdt, dydt]
#
#
# out = si.solve_ivp(ode_sys, np.array([0, 100]), np.array([1, 1]), method='RK45')

out = outRight

plt.plot(out[0, :], out[1, :], 'k')

print(np.shape(out))


plt.ylim([ymin, ymax])
plt.xlim([xmin, xmax])
# plt.hlines(0, xmin, xmax, 'k')
# plt.vlines(0, ymin, ymax, 'k')
plt.show()



# figu = go.Figure(
#     data=[go.Scatter(x=out[0,:],y=out[1,:])]
# )
# figu.update_layout(yaxis_range=[ymin, ymax])
# figu.update_layout(xaxis_range=[xmin, xmax])
#
# pio.renderers.default = "browser"
# figu.show()



# if __name__ == '__main__':
#     main()
