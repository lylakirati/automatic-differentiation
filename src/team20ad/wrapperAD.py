from .forwardAD import ForwardAD
from .reverseAD import ReverseAD


class AD:
    def __init__(self, var_dict, func_list):
        """
        Automatic Differentiation wrapper that automatically determines which mode to use.

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
        >>> ad = AD(var_dict, func_list)
        Number of variables <= number of functions; using forward mode.
        >>> ad()
        ===== Forward AD =====
        Vars: {'x': 1, 'y': 1}
        Funcs: ['x**2 + y**2', 'exp(x + y)']
        -----
        Func evals: [2, 7.38905609893065]
        Gradient:
        [[2.        2.       ]
        [7.3890561 7.3890561]]

        >>> var_dict = {'x': 1, 'y': 2, 'z': 3}
        >>> func_list = ['tan(x) + exp(y) + sqrt(z)']
        >>> ad = AD(var_dict, func_list)
        Number of variables > number of functions; using reverse mode.
        >>> ad()
        ===== Reverse AD =====
        Vars: {'x': 1, 'y': 2, 'z': 3}
        Funcs: ['tan(x) + exp(y) + sqrt(z)']
        -----
        Func evals: [10.67851463115443]
        Derivatives:
        [[3.42551882 7.3890561  0.28867513]]
        """
        use_forward = True
        if len(var_dict) <= len(func_list):
            print('Number of variables <= number of functions; using forward mode.')
        else:
            use_forward = False
            print('Number of variables > number of functions; using reverse mode.')

        if use_forward:
            res = ForwardAD(var_dict, func_list)
        else:
            res = ReverseAD(var_dict, func_list)

        self.res = res
        self.func_evals = res.func_evals
        self.Dpf = res.Dpf

    def __call__(self):
        return self.res.__call__()
