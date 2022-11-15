import numpy as np

from dualNumber import DualNumber


_supported_scalars = (int, float)


def sqrt(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.sqrt(val.real), 1 / 2 / np.sqrt(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.sqrt(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def exp(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.exp(val.real), np.exp(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.exp(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def log(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.log(val.real), 1 / val.real * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.log(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def sin(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.sin(val.real), np.cos(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.sin(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def cos(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.cos(val.real), -np.sin(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.cos(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def tan(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.tan(val.real), 1 / (np.cos(val.real) ** 2) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.tan(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def arcsin(val):
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
    if isinstance(val, DualNumber):
        return DualNumber(np.arctan(val.real), 1 / (1 + val.real ** 2) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.arctan(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def sinh(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.sinh(val.real), np.cosh(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.sinh(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def cosh(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.cosh(val.real), np.sinh(val.real) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.cosh(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")


def tanh(val):
    if isinstance(val, DualNumber):
        return DualNumber(np.tanh(val.real), (1 - (np.tanh(val.real) ** 2)) * val.dual)
    elif isinstance(val, _supported_scalars):
        return np.tanh(val)
    else:
        raise TypeError(f"Unsupported type '{type(val)}'")
