import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray



def runge_kutta(equation, domain: ndarray, seed, step_size):
    domain_length: int = len(domain)
    output: ndarray = np.zeros(domain_length)
    output[0] = seed
    for i in range(1, domain_length):
        c1 = step_size * equation(domain[i - 1], output[i - 1])
        c2 = step_size * equation(domain[i - 1] + .5 * step_size, output[i - 1] + .5 * c1)
        c3 = step_size * equation(domain[i - 1] + .5 * step_size, output[i - 1] + .5 * c2)
        c4 = step_size * equation(domain[i - 1] + step_size, output[i - 1] + c3)

        output[i] = output[i - 1] + ((1 / 6) * (c1 + 2 * c2 + 2 * c3 + c4))
    return output

# def runge_kutta_two_ways(equation, xmin, xmax, seed, step_size):
#
#     forward_domain_length: int = np.abs(xmax)/step_size
#     backwards_domain_length: int = np.abs(xmin)/step_size
#
#     out_forward = np.zeros(forward_domain_length)
#     out_backward = np.zeros(backwards_domain_length-1)
#
#     out_forward[0] = seed
#
#     step_size_back = -step_size
#
#     for i in range(1, max(out_forward, out_backward)):
#         if(i < forward_domain_length):
#             c1f = step_size * equation(domain[i - 1], out_forward[i - 1])
#             c2f = step_size * equation(domain[i - 1] + .5 * step_size, out_forward[i - 1] + .5 * c1f)
#             c3f = step_size * equation(domain[i - 1] + .5 * step_size, out_forward[i - 1] + .5 * c2f)
#             c4f = step_size * equation(domain[i - 1] + step_size, out_forward[i - 1] + c3f)
#             out_forward[i] = out_forward[i - 1] + ((1 / 6) * (c1f + 2 * c2f + 2 * c3f + c4f))
#         if(i < backwards_domain_length):
#             c1b = step_size_back * equation(domain[i - 1], out_backward[i - 1])
#             c2b = step_size_back * equation(domain[i - 1] + .5 * step_size_back, out_backward[i - 1] + .5 * c1b)
#             c3b = step_size_back * equation(domain[i - 1] + .5 * step_size_back, out_backward[i - 1] + .5 * c2b)
#             c4b = step_size_back * equation(domain[i - 1] + step_size_back, out_backward[i - 1] + c3b)
#             out_backward[i] = out_backward[i - 1] + ((1 / 6) * (c1b + 2 * c2b + 2 * c3b + c4b))
#
#     return output


def runge_kutta_any_order(equation_system, time_vector: ndarray, seeds: ndarray, step_size):
    """
    :param equation_system: An array containing the diff eq's to be solved for. WARNING: ORDER MATTERS.
    VERY MUCH. The first equation must be the lowest diff eq in the chain. For example, for a DHO, the
    first equation must be dx/dt = v, and the SECOND equation must be dv/dt = whatever
    :param time_vector: Time domain
    :param seeds: Initial starting values. MUST have size = order
    :param step_size: time step size
    :return: Numerical array of all outputs for x1, x2, ... xn:
    :author: Alex Carney
    """

    domain_length: int = len(time_vector)
    order = len(equation_system)
    output: ndarray = np.zeros((order, domain_length))
    output[:, 0] = seeds
    coefficients: ndarray = np.zeros((4, order))

    # For each time step (i), we do a few things:
    # For each equation in the system (f), we need to generate the required coefficients. The arguments factory
    # handles this, and populates the coefficients array for use with this current time step.
    # We start at 1 because index 0 are the seeds
    for i in range(1, domain_length):
        for j, f in enumerate(equation_system):
            for c_num in range(4):
                args = __function_arguments_factory(c_num, order, output, i, coefficients[:, j], time_vector, step_size)
                coefficients[c_num][j] = step_size * f(*args)
            output[j, i] = output[j, i - 1] + ((1 / 6) * (coefficients[0][j] + 2 * coefficients[1][j] + 2 * coefficients[2][j] + coefficients[3][j]))
    return output


def __function_arguments_factory(c_num: int, order: int, output: ndarray, i: int, required_coeff_array, time_vector: ndarray, step_size):
    """
    I discovered that if you pass in eqn(*args), including the asterisk, the equation will automatically
    unpack the arguments passed as a tuple and use them correctly. Sick.
    :param c_num: The coefficient to create. For 4th order RK, it can be between 0 and 3
    :param order: The order of the equation, the number of dependent variables - 1.
    :param i: current place in the rk domain
    :param required_coeff: The array of current coefficients being built up
    :return:
    """
    args = []
    # Add all dependent variables into the arguments list. From the table of coefficients
    for o in range(order):
        if c_num == 0:
            args.append(output[o, i - 1])
        elif c_num == 1:
            args.append(output[o, i - 1] + .5 * required_coeff_array[o])
        elif c_num == 2:
            args.append(output[o, i - 1] + .5 * required_coeff_array[o])
        elif c_num == 3:
            args.append(output[o, i - 1] + required_coeff_array[o])
    # Add all the independent variables (usually time) into the arguments list
    if c_num == 0:
        args.append(time_vector[i - 1])
    elif c_num == 1:
        args.append(time_vector[i - 1] + .5 * step_size)
    elif c_num == 2:
        args.append(time_vector[i - 1] + .5 * step_size)
    elif c_num == 3:
        args.append(time_vector[i - 1] + step_size)

    return args





def runge_kutta_second(equation_system, time_vector: ndarray, seeds: ndarray, step_size):
    domain_length: int = len(time_vector)
    output: ndarray = np.zeros((2, domain_length))
    output[:, 0] = seeds
    for i in range(1, domain_length):
        c11 = step_size * equation_system[0](output[0, i - 1], output[1, i - 1], time_vector[i - 1])
        c12 = step_size * equation_system[1](output[0, i - 1], output[1, i - 1], time_vector[i - 1])
        c21 = step_size * equation_system[0](output[0, i - 1] + .5 * c11, output[1, i - 1] + .5 * c12,
                                             time_vector[i - 1] + .5 * step_size)
        c22 = step_size * equation_system[1](output[0, i - 1] + .5 * c11, output[1, i - 1] + .5 * c12,
                                             time_vector[i - 1] + .5 * step_size)
        c31 = step_size * equation_system[0](output[0, i - 1] + .5 * c21, output[1, i - 1] + .5 * c22,
                                             time_vector[i - 1] + .5 * step_size)
        c32 = step_size * equation_system[1](output[0, i - 1] + .5 * c21, output[1, i - 1] + .5 * c22,
                                             time_vector[i - 1] + .5 * step_size)
        c41 = step_size * equation_system[0](output[0, i - 1] + c31, output[1, i - 1] + c32,
                                             time_vector[i - 1] + step_size)
        c42 = step_size * equation_system[1](output[0, i - 1] + c31, output[1, i - 1] + c32,
                                             time_vector[i - 1] + step_size)

        output[0, i] = output[0, i - 1] + ((1 / 6) * (c11 + 2 * c21 + 2 * c31 + c41))
        output[1, i] = output[1, i - 1] + ((1 / 6) * (c12 + 2 * c22 + 2 * c32 + c42))
    return output


def runge_kutta_2(equation_system, time_vector: ndarray, seeds: ndarray, step_size):
    domain_length: int = len(time_vector)
    output: ndarray = np.zeros((2, domain_length))
    output[:, 0] = seeds
    for i in range(1, domain_length):
        c11 = step_size * equation_system[0](output[0, i - 1], output[1, i - 1])
        c12 = step_size * equation_system[1](output[0, i - 1], output[1, i - 1])
        c21 = step_size * equation_system[0](output[0, i - 1] + .5 * step_size, output[1, i - 1] + .5 * c11)
        c22 = step_size * equation_system[1](output[0, i - 1] + .5 * step_size, output[1, i - 1] + .5 * c12)
        c31 = step_size * equation_system[0](output[0, i - 1] + .5 * step_size, output[1, i - 1] + .5 * c21)
        c32 = step_size * equation_system[1](output[0, i - 1] + .5 * step_size, output[1, i - 1] + .5 * c22)
        c41 = step_size * equation_system[0](output[0, i - 1] + step_size, output[1, i - 1] + c31)
        c42 = step_size * equation_system[1](output[0, i - 1] + step_size, output[1, i - 1] + c32)
        # c2 = step_size * equation(domain[i - 1] + .5 * step_size, output[j, i - 1] + .5 * c1)
        # c3 = step_size * equation(domain[i - 1] + .5 * step_size, output[j, i - 1] + .5 * c2)
        # c4 = step_size * equation(domain[i - 1] + step_size, output[j, i - 1] + c3)

        output[0, i] = output[0, i - 1] + ((1 / 6) * (c11 + 2 * c21 + 2 * c31 + c41))
        output[1, i] = output[1, i - 1] + ((1 / 6) * (c12 + 2 * c22 + 2 * c32 + c42))
        # output[j, i] = output[j, i - 1] + ((1 / 6) * (c1 + 2 * c2 + 2 * c3 + c4))
    return output


def runge_kutta_3(equation_system, time_vector: ndarray, seeds: ndarray, step_size):
    domain_length: int = len(time_vector)
    output: ndarray = np.zeros((2, domain_length))
    output[:, 0] = seeds
    for i in range(1, domain_length):
        output[0, i] = output[0, i - 1] + step_size * equation_system[0](output[0, i - 1], output[1, i - 1])
        output[1, i] = output[1, i - 1] + step_size * equation_system[1](output[0, i - 1], output[1, i - 1])
    return output


def hard_code(pos_eqn, vel_eqn, domain, x_seed, v_seed, step_size):
    domain_length: int = len(domain)
    print(domain_length)
    x = np.zeros(domain_length)
    v = np.zeros(domain_length)
    x[0] = x_seed
    v[0] = v_seed

    z = 1
    w0 = 3

    for i in range(1, domain_length):
        # print(str(vel_eqn(x[i-1], v[i-1])) + " VERSUS " + str((-2 * z * w0 * v[i-1]) - (x[i-1] * w0 ** 2)))
        # print("FOR SIMPLE" + str(pos_eqn(v[i-1])) + " VERSUS " + str(v[i-1]))

        IGNORE = 0
        v[i] = v[i - 1] + step_size * vel_eqn(x[i - 1], v[i - 1])
        x[i] = x[i - 1] + step_size * pos_eqn(_, v[i - 1])
        # v[i] = v[i-1] + step_size * ((-2 * z * w0 * v[i-1]) - (x[i-1] * w0 ** 2))
        # x[i] = x[i-1] + step_size * v[i-1]
    return x, v


if __name__ == '__main__':
    equation = lambda x, y: -y

    z = .25
    w0 = 10.917

    time_step = (1 / w0) * .001
    domain = np.r_[0:3: time_step]
    actual = 1 * np.exp(-domain)

    equation1 = lambda x, v: (-2 * z * w0 * v) - (x * w0 ** 2)
    equation2 = lambda x, v: v

    equation_system = np.array([equation2, equation1])

    # guessed_underdamped = runge_kutta_3(equation_system, domain, np.array([0, 0]), .1)

    # guessed_underdamped, _ = hard_code(equation2, equation1, domain, 5, 0, .1)

    # guessed_underdamped = runge_kutta_3(equation_system, domain, np.array([5, 0]), .1)
    guessed_underdamped = runge_kutta_2(equation_system, domain, np.array([1, 0]), time_step)
    # guessed_underdamped = runge_kutta_3([equation1, equation2], domain, np.array([5, 0]), .1)
    # better_guess = runge_kutta_3(equation_system, domain, np.array([1, 0]), time_step)

    A = 5
    equation1_td = lambda x, v, t: (-2 * z * w0 * v) - (x * w0 ** 2) + (A * t)
    equation2_td = lambda x, v, t: v

    new_attempt_guess = runge_kutta_second(np.array([equation2_td, equation1_td]), domain, np.array([1, 0]), time_step)

    # THIRD ORDER DIFF EQ TEST WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    eqn1 = lambda x, v, a, t: v
    eqn2 = lambda x, v, a, t: a
    eqn3 = lambda x, v, a, t: x

    t_step = .01
    time = np.r_[0:5:t_step]
    bruh = runge_kutta_any_order(np.array([eqn1, eqn2, eqn3]), time, np.array([1, 1, 1]), t_step)
    plt.plot(time, np.exp(time))
    plt.plot(time, bruh[0, :], 'r--')
    plt.show()

    #holy_crap = runge_kutta_any_order(np.array([equation2_td, equation1_td]), domain, np.array([1, 0]), time_step)

    # print(guessed_underdamped)
    # plt.plot(domain, guessed_underdamped[0, :], 'r')
    # #plt.plot(domain, holy_crap[0, :], 'b--')
    # # plt.plot(domain, new_attempt_guess[0, :], 'o')
    # # plt.plot(domain, better_guess[0, :], 'b--')
    # plt.show()

    # plt.plot(domain, guessed_underdamped[0, :])

    # guessed = runge_kutta(equation, domain, 1, time_step)

    #
    # plt.plot(domain, actual)
    # plt.plot(domain, guessed, 'r:')
    # plt.plot(domain, (actual - guessed) / actual)
    # plt.show()
