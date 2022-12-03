"""Docstring for the example.py module.

Modules names should have short, all-lowercase names.  The module name may
have underscores if this improves readability.

Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line.

"""

import numpy as np

from .dualNumber import DualNumber


_supported_scalars = (int, float)


def sqrt(val):
    """square root function supporting operations for forward mode AD.
    
    Parameter
    ------
    val : DualNumber, int or float
        value to compute square root
    """
    if isinstance(val, DualNumber):
        if val <= 0:
            raise ValueError(f"Should not be negative.")

        return DualNumber(np.sqrt(val.real), 1 / 2 / np.sqrt(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        if val <= 0:
            raise ValueError(f"Should not be negative.")

        return np.sqrt(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def exp(val):
    """exponential function (base natural) supporting operations for forward mode AD.
    
    Parameter
    ------
    val : DualNumber, int or float
        value to compute

    Notes
    ------
    exponential functions for other bases are handled by __pow__ in the DualNumber class.
    """
    if isinstance(val, DualNumber):
        return DualNumber(np.exp(val.real), np.exp(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.exp(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def log(val, base = np.exp(1)):
    """Logarithmic function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute the log
    base : int or float
        base value of log function, optional (default = natural e)
    """
    if isinstance(val, DualNumber):
        if val.real <= 0:
            raise ValueError(f"Should not be negative.")
        real = np.log(val.real) / np.log(base)
        dual = (1 / val.real / np.log(base)) * val.dual
        return DualNumber(real, dual)
    elif isinstance(val, _supported_scalars):
        if val <= 0:
            raise ValueError(f"Should not be negative.")
        return np.log(val) / np.log(base)
    else: 
        raise TypeError(f"Unsupported type '{type(val)}'")


def sin(val):
    """Sine function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute sine
    """
    if isinstance(val, DualNumber):
        return DualNumber(np.sin(val.real), np.cos(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.sin(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def cos(val):
    """Cosine function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute cosine
    """
    if isinstance(val, DualNumber):
        return DualNumber(np.cos(val.real), -np.sin(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.cos(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def tan(val):
    """Tangent function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute tangent
    """
    if isinstance(val, DualNumber):
        x = val.real % np.pi == (np.pi / 2)
        if x:
            raise ValueError('Tan is undefined in the given domain')

        return DualNumber(np.tan(val.real), 1 / (np.cos(val.real) ** 2) * val.dual)
    elif isinstance(val, _supported_scalars):
        x = val.real % np.pi == (np.pi / 2)
        if x:
            raise ValueError('Tan is undefined in the given domain')

        return np.tan(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def arcsin(val):
    """Inverse sine function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute inverse sine
    """
    if isinstance(val, DualNumber):
        if abs(val.real) >= 1:
            raise ValueError(
                'arcsin() cannot be evaluated at {}.'.format(val.real))
        return DualNumber(np.arcsin(val.real), 1 / np.sqrt(1 - val.real ** 2) * val.dual)
    elif isinstance(val, _supported_scalars):
        if abs(val) >= 1:
            raise ValueError('arcsin() cannot be evaluated at {}.'.format(val))
        return np.arcsin(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def arccos(val):
    """Inverse cosine function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute inverse cosine
    """
    if isinstance(val, DualNumber):
        if abs(val.real) >= 1:
            raise ValueError(
                'arccos() cannot be evaluated at {}.'.format(val.real))
        return DualNumber(np.arccos(val.real), -1 / np.sqrt(1 - val.real ** 2) * val.dual)
    elif isinstance(val, _supported_scalars):
        if abs(val) >= 1:
            raise ValueError('arccos() cannot be evaluated at {}.'.format(val))
        return np.arccos(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def arctan(val):
    """Inverse tangent function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute inverse tangent
    """
    if isinstance(val, DualNumber):
        return DualNumber(np.arctan(val.real), 1 / (1 + val.real ** 2) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.arctan(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def sinh(val):
    """Hyperbolic sine function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute hyerbolic sine
    """
    if isinstance(val, DualNumber):
        return DualNumber(np.sinh(val.real), np.cosh(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.sinh(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def cosh(val):
    """Hyperbolic cosine function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute hyerbolic cosine
    """
    if isinstance(val, DualNumber):
        return DualNumber(np.cosh(val.real), np.sinh(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.cosh(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def tanh(val):
    """Hyperbolic tangent function supporting operations for forward mode AD.

    Parameter
    ------
    val : DualNumber, int or float
        value to compute hyerbolic tangent
    """
    if isinstance(val, DualNumber):
        return DualNumber(np.tanh(val.real), (1 - (np.tanh(val.real) ** 2)) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.tanh(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")
