import numpy as np
import re

from .elementary import *


class ReverseAD:
    def __init__(self, var_dict, func_list):
        """
        Reverse Mode Automatic Differentiation.
        Inputs
        ------
        var_dict: dict
            a dictionary of variables and their corresponding values
        func_list: str or list of str
            (a list of) function(s) encoded as string(s)

        Examples
        --------
        >>> var_dict = {'x': 1, 'y': 1}
        >>> func_list = ['x**2 + y**2', 'exp(x + y)']
        >>> ad = ReverseAD(var_dict, func_list)

        ===== Reverse AD =====

        Vars: {'x': 1, 'y': 1}

        Funcs: ['x**2 + y**2', 'exp(x + y)']

        -----

        Func evals: [2, 7.38905609893065]

        Gradient:

        [[2.        2.       ]
        [7.3890561 7.3890561]]        
        """
        # type checks
        if not isinstance(var_dict, dict):
            raise TypeError("var_dict should be a dictionary.")

        if isinstance(func_list, list):
            for f in func_list:
                if not isinstance(f, str):
                    raise TypeError(
                        "func_list should be a string or a list of strings.")
        elif not isinstance(func_list, str):
            raise TypeError(
                "func_list should be a string or a list of strings.")

        if isinstance(func_list, list):
            self.func_list = func_list
        else:
            self.func_list = [func_list]

        self.func_evals = []
        self.Dpf = []
        self.var_dict = var_dict

        elem_funcs = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
                      'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']

        for func in self.func_list:
            for i in elem_funcs:
                if i in func:
                    func = re.sub(i + r'\(', 'Node.' + i + '(', func)
                    func = re.sub('arcNode.', 'arc', func)

            for var_name, var_value in var_dict.items():
                exec(f'{var_name} = Node(float(var_value))')
            vals = eval(func)

            value_keys = str(list(var_dict.keys())).replace('\'', '')
            v, d = eval(f'vals.g_derivatives({value_keys})')

            self.func_evals.append(v)
            self.Dpf.append(d)
        self.Dpf = np.array(self.Dpf)

    def __call__(self):
        out = "===== Forward AD =====\n"
        out += f"Vars: {self.var_dict}\n"
        out += f"Funcs: {self.func_list}\n"
        out += f"-----\n"
        out += f"Func evals: {self.func_evals}\n"
        out += f"Derivatives:\n{self.Dpf}\n"
        print(out)


class Node():
    def __init__(self, var):
        """
        Node object with follow attributes:
        child: a list of all depending Nodes and derivatives
        derivative: Representing evaluated derivatives
        """
        if isinstance(var, int) or isinstance(var, float):
            self.var = var
            self.child = []
            self.derivative = None
        else:
            raise TypeError("Input is not a real number.")

    def g_derivatives(self, inputs):
        """
        Get derivatives for each variable in the function
        var_val: a variable which stores the function values
        der_list: a list of derivatives for each variable

        """
        # self.der = 1
        var_val = self.var
        der_list = np.array([var_i.partial() for var_i in inputs])
        return var_val, der_list

    def partial(self):
        """
        Computes derivative for a variable used in the function
        """
        if len(self.child) == 0:
            return 1
        if self.derivative is not None:
            return self.derivative
        else:
            self.derivative = sum(
                [child.partial() * partial for child, partial in self.child])
            return self.derivative

    def __add__(self, other):

        try:
            new_add = Node(self.var + other.var)
            self.child.append((new_add, 1))
            other.child.append((new_add, 1))

            return new_add
        except:
            if isinstance(other, int) or isinstance(other, float):
                new_add = Node(self.var + other)
                self.child.append((new_add, 1))
                return new_add
            else:
                raise TypeError("Not real number")

    def __mul__(self, other):
        try:
            new_mul = Node(other.var * self.var)
            self.child.append((new_mul, other.var))
            other.child.append((new_mul, self.var))
            return new_mul
        except:
            if isinstance(other, int) or isinstance(other, float):
                # other is not a Node and the multiplication could complete if it is a real number
                new_mul = Node(other * self.var)
                self.child.append((new_mul, other))
                return new_mul
            else:
                raise TypeError("Input is not a real number.")

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __truediv__(self, other):
        try:
            new_div = Node(self.var / other.var)
            self.child.append(
                (new_div, ((1 * other.var - 0 * self.var) / other.var**2)))
            other.child.append((new_div, (-self.var/(other.var**2))))
            return new_div
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_div = Node(self.var / other)
                self.child.append(
                    (new_div, ((1 * other - 0 * self.var) / other**2)))
                return new_div
            else:
                raise TypeError(f"{other} is invalid.")

    def __neg__(self):
        new_neg = Node(-self.var)
        self.child.append((new_neg, -1))
        return new_neg

    def __rtruediv__(self, other):
        try:
            new_div = Node(other.var / self.var)
            self.child.append(
                (new_div, ((0 * self.var - other.var * 1) / self.var**2)))
            other.child.append((new_div, 1/self.var))
            return new_div
        except:
            if isinstance(other, int) or isinstance(other, float):
                new_div = Node(other / self.var)
                self.child.append(
                    (new_div, ((0 * self.var - other * 1) / self.var**2)))
                return new_div
            else:
                raise TypeError(f"Input {other} is not valid.")

    def __lt__(self, other):
        try:
            return self.var < other.var
        except AttributeError:
            return self.var < other

    def __gt__(self, other):
        try:
            return self.var > other.var
        except AttributeError:
            return self.var > other

    def __le__(self, other):
        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= other

    def __ge__(self, other):
        try:
            return self.var >= other.var
        except AttributeError:
            return self.var >= other

    def __eq__(self, other):
        try:
            return self.var == other.var
        except:
            raise TypeError('Input incomparable')

    def __ne__(self, other):
        return not self.__eq__(other)

    def __abs__(self):
        new_abs = Node(abs(self.var))
        self.child.append((1, new_abs))
        return new_abs

    def __pow__(self, other):
        try:
            new_val = Node(self.var ** other.var)
            self.child.append(
                (new_val, (other.var) * self.var ** (other.var-1)))
            other.child.append(
                (new_val, self.var ** other.var * (np.log(self.var))))
            return new_val
        except:
            if isinstance(other, int):
                new_val = Node(self.var ** other)
                self.child.append((new_val, (other) * self.var ** (other-1)))
                return new_val
            else:
                raise TypeError(f"Exponent is invalid.")

    def __rpow__(self, other):
        try:
            new_val = Node(other ** self.var)
        except:
            raise ValueError("must be a number.")
        self.child.append((new_val, other**self.var * np.log(other)))
        return new_val

    @staticmethod
    def log(var):
        try:
            if var.var <= 0:
                raise ValueError('Input must to be greater than 0.')
        except:
            raise TypeError(f"Invalid input")
        log_var = Node(np.log(var.var))
        var.child.append((log_var, (1. / var.var) * 1))
        return log_var

    @staticmethod
    def sqrt(var):
        if var < 0:
            raise ValueError("Invalid input")
        else:
            try:
                sqrt_var = Node(var.var**(1/2))
                var.child.append((sqrt_var, (1/2)*var.var**(-1/2)))
            except:
                raise TypeError(f"Invalid input")
        return sqrt_var

    @staticmethod
    def exp(var):
        try:
            new_val = Node(np.exp(var.var))
            var.child.append((new_val, np.exp(var.var) * 1))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Invalid input")

            return np.exp(var)

    @staticmethod
    def sin(var):
        try:
            new_val = Node(np.sin(var.var))
            var.child.append((new_val, 1 * np.cos(var.var)))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Invalid input")

            return np.sin(var)

    @staticmethod
    def cos(var):

        try:
            new_val = Node(np.cos(var.var))
            var.child.append((new_val, 1 * -np.sin(var.var)))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Invalid input")

            return np.cos(var)

    @staticmethod
    def tan(var):
        try:
            new_val = Node(np.tan(var.var))
            var.child.append((new_val, 1 * 1 / np.power(np.cos(var.var), 2)))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")

            return np.tan(var)

    @staticmethod
    def arcsin(var):
        try:
            if var.var > 1 or var.var < -1:
                raise ValueError('Please input -1 <= x <=1')
            else:
                new_val = Node(np.arcsin(var.var))
                var.child.append((new_val, 1 / np.sqrt(1 - (var.var ** 2))))
                return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
            return np.arcsin(var)

    @staticmethod
    def arccos(var):
        try:
            if isinstance(var, int) or isinstance(var, float):
                return np.arccos(var)

            if var.var > 1 or var.var < -1:
                raise ValueError('Please input -1 <= x <=1')
            else:
                new_val = Node(np.arccos(var.var))
                var.child.append((new_val, -1 / np.sqrt(1 - (var.var ** 2))))
            return new_val
        except:
            raise TypeError(f"Input {var} is not valid.")

    @staticmethod
    def arctan(var):
        try:
            new_val = Node(np.arctan(var.var))
            var.child.append((new_val, 1 * 1 / (1 + np.power(var.var, 2))))

            return new_val

        except AttributeError:
            return np.arctan(var)

    @staticmethod
    def sinh(var):
        try:
            new_val = Node(np.sinh(var.var))
            var.child.append((new_val, 1 * np.cosh(var.var)))
            return new_val

        except AttributeError:
            return np.sinh(var)

    @staticmethod
    def cosh(var):
        try:
            new_val = Node(np.cosh(var.var))
            var.child.append((new_val, 1 * np.sinh(var.var)))

            return new_val

        except AttributeError:
            return np.cosh(var)

    @staticmethod
    def tanh(var):
        try:
            new_val = Node(np.tanh(var.var))
            var.child.append((new_val, 1 * 1 / np.power(np.cosh(var.var), 2)))
            return new_val
        except AttributeError:
            return np.tanh(var)

    def sigmoid(var):
        try:
            logistic_var = Node(1 / (1 + np.exp(-var.var)))
            var.child.append((logistic_var, 1 / (1 + np.exp(-var.var))
                             * (1-(1 / (1 + np.exp(-var.var)) * 1))))
            return logistic_var
        except:
            raise TypeError(f"Invalid input")

    def __str__(self):
        return f"value = {self.var}\n derivative = {self.partial()}"

    def __repr__(self):
        return f"value = {self.var}\n derivative = {self.partial()}"
