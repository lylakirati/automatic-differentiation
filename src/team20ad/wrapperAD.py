from .forwardAD import ForwardAD
from .reverseAD import ReverseAD


class AD:
    def __init__(self, var_dict, func_list, mode = None):
        """
        Automatic Differentiation wrapper that a mode can be specified. 
        If not, it automatically determines which mode to use based on the 
        number of independent variables and the number of functions to differentiate.

        Inputs
        ------
        var_dict: dict
            a dictionary of variables and their corresponding values
        func_list: str or list of str
            (a list of) function(s) encoded as string(s)
        mode: str, optional (default = None)
            string indicating mode of AD. Can be either "forward"/"f"/"reverse"/"r"

        Examples
        --------
        >>> var_dict = {'x': 1, 'y': 1}
        >>> func_list = ['x**2 + y**2', 'exp(x + y)']
        >>> ad = AD(var_dict, func_list)
        Number of variables <= number of functions: forward mode by default.
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
        Number of variables > number of functions: reverse mode by default.
        >>> ad()
        ===== Reverse AD =====
        Vars: {'x': 1, 'y': 2, 'z': 3}
        Funcs: ['tan(x) + exp(y) + sqrt(z)']
        -----
        Func evals: [10.67851463115443]
        Derivatives:
        [[3.42551882 7.3890561  0.28867513]]
        """
        # check mode param valid
        if (mode is not None) and (mode not in ("forward", "f", "reverse", "r")):
            raise ValueError(f"Mode can be either forward, f, reverse, r, or None.") 
        
        self.mode = mode
        if self.mode is None:
            if len(var_dict) <= len(func_list):
                self.mode = "forward"
                print('Number of variables <= number of functions: forward mode by default.')
            else:
                self.mode = "reverse"
                print('Number of variables > number of functions: reverse mode by default.')

        if self.mode in ("forward", "f"):
            self.res = ForwardAD(var_dict, func_list)
        else:
            self.res = ReverseAD(var_dict, func_list)

        self.func_evals = self.res.func_evals
        self.Dpf = self.res.Dpf

    def __call__(self):
        return self.res.__call__()
