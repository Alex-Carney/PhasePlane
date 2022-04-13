import numpy as np
import matplotlib.pyplot as plt
from util import carney_diff_eqs as ode44

# def main() -> None:
eqn1 = lambda x, y, t: x
eqn2 = lambda x, y, t: y

time_step_size: float = .1
h = -.1

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

out = ode44.runge_kutta_any_order(
    np.array([eqn1, eqn2]),
    time,
    np.array([1, 1]),
    h)

plt.plot(out[0, :], out[1, :], 'k')


plt.ylim([ymin, ymax])
plt.xlim([xmin, xmax])
plt.hlines(0, xmin, xmax, 'k')
plt.vlines(0, ymin, ymax, 'k')
plt.show()


# if __name__ == '__main__':
#     main()
