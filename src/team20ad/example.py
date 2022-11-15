from forwardAD import *


if __name__ == '__main__':
    # pp7 example
    v = {'x': 1, 'y': 1}
    f = ['x**2 + y**2', 'exp(x + y)']
    # Analytical solution: J(1, 1) = [[2, 2], [e^2, e^2]]

    ad = ForwardAD(v, f)
    ad()
