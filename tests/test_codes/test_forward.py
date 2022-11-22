import sys
sys.path.append("./src/")

import pytest
import numpy as np
from team20ad.dualNumber import DualNumber
from team20ad.elementary import *
from team20ad.forwardAD import *


class TestForwardAD:

    def test_initializer(self):
        with pytest.raises(TypeError):
            x = DualNumber("hello")

    def test_add_radd(self):
        x = DualNumber(3, 1)
        y = x + 3

        assert y.real == 6
        assert y.dual == 1

        x = DualNumber(3, 1)
        y = x + DualNumber(3, 1)

        assert y.real == 6
        assert y.dual == 2


    def test_mul_rmul(self):
        x = DualNumber(3, 1)
        y = x * 3

        assert y.real == 9
        assert y.dual == 3

        x = DualNumber(3)
        y = x * DualNumber(3)

        assert y.real == 9
        assert y.dual == 6

        x = DualNumber(3, 1)
        y = x * DualNumber(3, 0)

        assert y.real == 9
        assert y.dual == 3

        x = DualNumber(3, 1)
        y = DualNumber(3, 0) * x

        assert y.real == 9
        assert y.dual == 3

    def test_sub_rsub(self):
        x = DualNumber(3, 1)
        y = x - 3

        assert y.real == 0
        assert y.dual == 1

        x = DualNumber(3, 1)
        y = x - DualNumber(3, 1)

        assert y.real == 0
        assert y.dual == 0

        x = DualNumber(4, 1)
        y = x - DualNumber(3, 0)

        assert y.real == 1
        assert y.dual == 1

        x = DualNumber(3, 1)
        y = 3 - x

        assert y.real == 0
        assert y.dual == -1

        x = DualNumber(3, 1)
        y = DualNumber(3, 1) - x

        assert y.real == 0
        assert y.dual == 0

        x = DualNumber(3, 1)
        y = DualNumber(4, 0) - x

        assert y.real == 1
        assert y.dual == -1

    def test_truediv_rtruediv(self):
        x = DualNumber(3, 1)
        y = x / 3

        assert y.real == 1
        assert y.dual == 1 / 3

        x = DualNumber(3, 1)
        y = x / DualNumber(4, 1)

        assert y.real == 3 / 4
        assert y.dual == 1 / 16

        x = DualNumber(3, 1)
        y = x / DualNumber(4, 0)

        assert y.real == 3 / 4
        assert np.array_equal(y.dual, 1 / 4)

        x = DualNumber(3, 1)
        y = 2 / x

        assert y.real == 2 / 3
        assert y.dual == -2 / 9

        x = DualNumber(4, 1)
        y = DualNumber(3, 1) / x

        assert y.real == 3 / 4
        assert y.dual == 1 / 16

        x = DualNumber(4, 0)
        y = DualNumber(3, 1) / x

        assert y.real == 3 / 4
        assert np.array_equal(y.dual, 1 / 4)

    def test_neg(self):
        x = DualNumber(3, 1)
        y = -x
        assert y.real == -3
        assert y.dual == -1


    def test_eq(self):
        X = DualNumber(3, 1)
        Y = DualNumber(3, 1)
        flag = (X == Y)
        assert flag == True

        flag = (DualNumber(3, 1) == 3)
        assert flag == False

    def test_ne(self):
        X = DualNumber(3, 1)
        Y = DualNumber(3, 1)
        flag = (X != Y)
        assert flag == False

        flag = (DualNumber(3, 1) != 3)
        assert flag == True



    def test_lt(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X < Y)
        assert flag == True

        flag = (DualNumber(3, 1) < 3)
        assert flag == False
        
        
    def test_le(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X < Y)
        assert flag == True

        flag = (DualNumber(3, 1) < 3)
        assert flag == False

    def test_gt(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X > Y)
        assert flag == False

        flag = (DualNumber(3, 1) > 3)
        assert flag == False

    def test_ge(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X >= Y)
        assert flag == False

        flag = (DualNumber(3, 1) >= 3)
        assert flag == True

    def test_abs(self):
        y = abs(DualNumber(-3, -1))
        assert y.real == 3
        assert y.dual == 1







    def test_pow(self):
        # test DualNumber raise to a constant
        x = DualNumber(5)
        y = x ** 3

        assert y.real == 5 ** 3
        assert y.dual == 3 * (5 ** 2)

    def test_pow_var(self):
        # test DualNumber raise to a DualNumber
        x = DualNumber(2)
        y = x ** (3 * x)

        assert y.real == 2 ** 6
        assert y.dual == 192 + 192 * np.log(2)

    def test_pow_imaginary(self):
        # test pow when base < 0 and exponent < 1
        with pytest.raises(TypeError):
            x = DualNumber(-1)
            y = x ** -0.5

    def test_rpow(self):
        # test constant raise to a DualNumber
        x = DualNumber(3)
        y = 5 ** x
        assert y.real == 5 ** 3
        assert y.dual == 125 * np.log(5)

    def test_rpow_imaginary(self):
        # test constant raise to a DualNumber
        with pytest.raises(TypeError):
            x = DualNumber(0.5)
            y = (-2) ** x





    def test_sqrt(self):
        # test square root of a DualNumber
        x = DualNumber(10.1)
        y = sqrt(x)
        assert y.real == np.sqrt(10.1)
        assert y.dual == 1 / (2 * (np.sqrt(10.1)))

    def test_sqrt_constant(self):
        # test square root of a DualNumber
        x = 12
        y = sqrt(x)
        assert y == np.sqrt(12)

    def test_sqrt_non_positive(self):
        # test square root of a non-positve DualNumber
        with pytest.raises(ValueError):
            x = DualNumber(-10.1)
            y = sqrt(x)

    def test_exp(self):
        x = DualNumber(32)
        y = exp(x)
        assert y.real == np.exp(32)
        assert y.dual == np.exp(32)

    def test_exp_constant(self):
        x = 32
        y = exp(x)
        assert y == np.exp(32)

    def test_log(self):
        x = DualNumber(14)
        y = log(x)
        assert y.real == np.log(14)
        assert y.dual == 1 / 14

    def test_log_constant(self):
        x = 14
        y = log(x)
        assert y == np.log(14)

    def test_log_non_positive(self):
        with pytest.raises(ValueError):
            x = DualNumber(-14)
            y = log(x)

    def test_tangent_function(self):
        x = DualNumber(np.pi)
        f = tan(x)

        # However, if you specify a message with the assertion like this:
        # assert a % 2 == 0, "value was odd, should be even"
        # then no assertion introspection takes places at all and the message will be simply shown in the traceback.
        assert f.dual == 1

        x = DualNumber(3 * np.pi / 2)
        with pytest.raises(ValueError):
            f = tan(x)

        x = DualNumber(2)
        f = 3 * tan(x)

        assert f.real == 3 * np.tan(2) and np.round(f.dual, 4) == 17.3232

        x = DualNumber(np.pi)
        f = tan(x) * tan(x)

        assert np.round(f.dual, 5) == 0

        # checking a constant
        assert tan(3) == np.tan(3)

    # can use these function below to run the code manually rather than with pytest
    def test_arctangent_function(self):
        x = DualNumber(2)
        f = arctan(x)

        assert f.dual == .2 and np.round(f.real, 4) == 1.1071

        # check a constant
        assert arctan(3) == np.arctan(3)

    def test_sinh_function(self):
        x = DualNumber(2)
        f = 2 * sinh(x)

        assert np.round(f.dual, 4) == 7.5244 and np.round(f.real, 4) == 7.2537

        # check a constant
        assert sinh(3) == np.sinh(3)

    def test_cosh_function(self):
        x = DualNumber(4)
        f = 3 * cosh(x)
        assert np.round(f.real, 4) == 81.9247 and np.round(f.dual, 4) == 81.8698

        # check a constant
        assert cosh(3) == np.cosh(3)

    def test_tanh_function(self):
        x = DualNumber(3)
        f = 2 * tanh(x)
        assert np.round(f.real, 4) == 1.9901 and np.round(f.dual, 4) == 0.0197

        # checking a constant
        assert tanh(3) == np.tanh(3)

    def test_sin(self):
        x = DualNumber(0)
        f = sin(x)

        assert f.real == 0.0
        assert f.dual == 1.

        # check constant
        assert sin(2) == np.sin(2)

    def test_cos(self):
        x = DualNumber(0)
        f = cos(x)
        assert f.real == 1.0
        assert f.dual == 0.

        # check constant
        assert cos(2) == np.cos(2)

    def test_arcsin(self):
        x = DualNumber(0)
        f = arcsin(x)
        assert f.real == 0.0
        assert f.dual == 1.
        # -1<= x <=1

        with pytest.raises(ValueError):
            x = DualNumber(-2)
            f = arcsin(x)

        assert arcsin(0.5) == np.arcsin(0.5)

    def test_arccos(self):
        x = DualNumber(0)
        f = arccos(x)
        assert np.round(f.real, 4) == 1.5708
        assert f.dual == -1.

        with pytest.raises(ValueError):
            x = DualNumber(2)
            f = arccos(x)

        assert arccos(0.5) == np.arccos(0.5)

   

    def test_ForwardAD(self):
        vars = {'x': 0.5, 'y': 4}
        fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3', 'sqrt(x)/3', '3 * sinh(x) - 4 * arcsin(x) + 5']
        z = ForwardAD(vars, fcts)

        assert np.array_equal(np.around(z.func_evals, 4), np.array([16.8776, 2.5369, 0.2357, 4.4689]))
        assert np.array_equal(np.around(z.Dpf, 4),
                              np.array([[-0.4794, 8.], [-0.2357, 0.5], [0.2357, 0.], [-1.2359, 0.]]))




    def test_repr_str(self):
        vars = {'x': 0.5, 'y': 4}
        fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3', 'sqrt(x)/3', '3 * sinh(x) - 4 * arcsin(x) + 5']
        z = ForwardAD(vars, fcts)
        assert isinstance(z.__str__(), str)
        assert isinstance(z.__repr__(), str)

