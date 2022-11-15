import numpy as np


class DualNumber:

    _supported_scalars = (int, float)

    def __init__(self, real, dual=1.0):
        if isinstance(real, self._supported_scalars):
            self.real = real
            self.dual = dual
        else:
            raise TypeError("Supported scalars: {_supported_scalars}")

    def __repr__(self):
        return f"DualNumber({self.real}, {self.dual})"

    def __str__(self):
        return f"DualNumber: real = {self.real}, dual = {self.dual}"

    def __neg__(self):
        return DualNumber(-self.real, -self.dual)

    def __add__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real + other, self.dual)
        return DualNumber(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return DualNumber(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        return DualNumber(other.real - self.real, other.dual - self.dual)

    def __mul__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real * other, other * self.dual)
        return DualNumber(self.real * other.real,
                          self.real * other.dual + other.real * self.dual)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return DualNumber(self.real / other, self.dual / other)
        if other.real == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return DualNumber(self.real / other.real,
                          (self.other * other.real - self.real * other.dual) / (other.real ** 2))

    def __rtruediv__(self, other):
        if not isinstance(other, self._supported_scalars):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return DualNumber(other / self.real, (-other / self.real ** 2) * self.dual)

    def __pow__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            real_pow = self.real ** other
            dual_pow = other * (self.real ** (other - 1)) * self.dual
        else:
            if self.real > 0:
                real_pow = self.real ** other.real
                dual_pow = other.real * self.real ** (other.real - 1) * self.dual + np.log(
                    self.real) * self.real ** other.real * other.dual
            else:
                real_pow = self.real ** other.real
                dual_pow = other.real * \
                    self.real ** (other.real - 1) * self.dual
        return DualNumber(real_pow, dual_pow)

    def __rpow__(self, other):
        if not isinstance(other, self._supported_scalars):
            raise TypeError(f"Unsupported type '{type(other)}'")
        real_pow = other ** self.real
        dual_pow = np.log(other) * other ** self.real * self.dual
        return DualNumber(real_pow, dual_pow)

    def __eq__(self, other):
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return (self.real == other.real) and (self.dual == 0)
        return (self.real == other.real) and (self.dual == other.dual)

    def __ne__(self, other):
        return not self.__eq__(other)
