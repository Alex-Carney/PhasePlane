import numpy as np

"""
Weierstrass root finding method. Finds all roots at once, starting from a complex seed value. 

@author: Alex Carney
"""


def roots(polynomial, equation_order, seed_value: complex = (2 + 1j), tolerance: float = .01):
    """
    :param equation_order: order of the input polynomial
    :param tolerance: User defined accuracy
    :param polynomial: A polynomial equation of type np.poly1d
    :param seed_value: An initial starting value, a complex number a where |a| != 1
    :return: A list of roots
    """
    # input validation
    if np.abs(seed_value) == 1:
        print("Invalid seed value, cannot have absolute value = 1")
        return
    else:
        roots_array: np.ndarray = np.zeros(equation_order, dtype=complex)
        # initialize each starting root to a^i
        for i in range(len(roots_array)):
            roots_array[i] = seed_value ** i
        return _roots_helper(polynomial, roots_array, tolerance, 0)


def _roots_helper(polynomial, roots_array: np.ndarray, tolerance, depth):
    depth = depth + 1
    # check recurrence escape condition
    if ((np.abs(polynomial(roots_array))) <= tolerance).all():
        print("Found all roots at a depth of " + str(depth))
        return roots_array
    else:
        return _roots_helper(polynomial,
                             (roots_array - (polynomial(roots_array) / modified_product(roots_array))),
                             tolerance,
                             depth)


def modified_product(roots_array: np.ndarray):
    """
    Computes the product, with the exception of ignoring the case where i = j.
    :return:
    """
    n_max = len(roots_array)
    product_list = np.zeros(n_max, dtype=complex)
    for j in range(n_max):
        running_product = np.zeros(n_max, dtype=complex)
        for i in range(n_max):
            running_product[i] = roots_array[j] - roots_array[i] if i != j else 1 + 0j
        product_list[j] = running_product.prod()
    return product_list


def main():
    # peqn = np.poly1d([1, 0, 0, -2, 0, -3])
    peqn = lambda x: x ** 5 - 2 * x ** 2 - 3
    # peqn = np.poly1d([1, -5, 6])
    # print("Should be " + str(peqn.roots))
    r = roots(peqn, 5, (2 + 1j), .01)
    print(str(r))


if __name__ == '__main__':
    main()
