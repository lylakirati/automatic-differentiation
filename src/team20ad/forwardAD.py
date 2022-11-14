import numpy as np

from elementary import *


class ForwardAD:

    def __init__(self, var_dict, func_list):
        """
        Forward Mode Automatic Differentiation.

        Inputs
        ------
        var_dict: a dictionary of variables and their corresponding values
        func_list: a list of functions encoded as strings

        Returns
        -------

        Examples
        --------
        >>> var_dict = {'x': 1, 'y': 1}
        >>> func_list = ['x**2 + y**2', 'exp(x + y)']
        >>> ad = ForwardAD(var_dict, func_list)

        ===== Forward AD =====

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
            raise TypeError("var_dict should be a dictionary")
        for f in func_list:
            if not isinstance(f, str):
                raise TypeError("func_list should be a list of strings")

        # var inits
        elem_funcs = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
                      'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']
        self.var_dict = var_dict
        self.func_list = func_list
        vars = list(var_dict.keys())

        self.func_evals = []
        self.Dpf = np.zeros((len(func_list), len(var_dict)))
        i = 0  # a helper counter to determine which partial deriv to take

        for _ in var_dict:
            for var in var_dict:
                if var == vars[i]:
                    dual = 1
                else:
                    dual = 0
                exec(f"{var} = DualNumber({var_dict[var]}, {dual})")

            for j in range(0, len(func_list)):
                for f in elem_funcs:
                    func = func_list[j]
                    if f in func_list[j]:
                        break
                self.func_evals.append(eval(func).real)  # primal trace
                self.Dpf[j, i] = eval(func).dual  # tangent trace

            i += 1

    def __repr__(self):
        out = "===== Forward AD =====\n"
        out += f"Vars: {self.var_dict}\n"
        out += f"Funcs: {self.func_list}\n"
        out += f"-----\n"
        out += f"Func evals: {self.func_evals[:len(self.func_list)]}\n"
        out += f"Gradient:\n"
        out += f"{self.Dpf}"
        return out

    def __str__(self):
        out = "===== Forward AD =====\n"
        out += f"Vars: {self.var_dict}\n"
        out += f"Funcs: {self.func_list}\n"
        out += f"-----\n"
        out += f"Func evals: {self.func_evals[:len(self.func_list)]}\n"
        out += f"Gradient:\n"
        out += f"{self.Dpf}"
        return out
