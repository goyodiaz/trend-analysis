#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Goyo <goyodiaz@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.


import math
import unittest

import numpy as np
import pymannkendall as mk
from trend import TrendTestResult, trend_test


class Test_trend_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(seed=0)
        cls.data = rng.normal(loc=5.2, scale=0.6, size=100)

    def test_linear_test(self):
        result = trend_test(self.data, kind="linear")
        assert_is_close(0.0008823394604103146, result.slope)
        assert_is_close(5.2049822128041185, result.intercept)
        assert_is_close(0.6629324836174293, result.pvalue)

    def test_mk_original_test(self):
        result = trend_test(self.data, kind="mann-kendall", mk_test=mk.original_test)
        assert_is_close(0.0006616891887145094, result.slope)
        assert_is_close(5.21105876676486, result.intercept)
        assert_is_close(0.7499854902791532, result.pvalue)

    def test_mk_seaonal_test(self):
        result = trend_test(self.data, kind="mann-kendall", mk_test=mk.seasonal_test)
        assert_is_close(0.025424323693072938, result.slope)
        assert_is_close(5.138937046372303, result.intercept)
        assert_is_close(0.4816457158358991, result.pvalue)

    def test_mk_correlated_seaonal_test(self):
        result = trend_test(
            self.data, kind="mann-kendall", mk_test=mk.correlated_seasonal_test
        )
        assert_is_close(0.025424323693072938, result.slope)
        assert_is_close(5.138937046372303, result.intercept)
        assert_is_close(0.13239875702917048, result.pvalue)

    def test_wrapped(self):
        result = trend_test(self.data, kind="linear")
        assert_is_close(0.0008823394604103141, result.wrapped.slope)
        assert_is_close(5.2049822128041185, result.wrapped.intercept)
        assert_is_close(0.6629324836174293, result.wrapped.pvalue)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            result = trend_test(self.data, kind="mxyzptlk")


class TestTrendTestResult(unittest.TestCase):
    def test_trend(self):
        result = TrendTestResult(slope=0, intercept=0, pvalue=0.24)
        self.assertTrue(result.trend(alpha=0.25))
        self.assertFalse(result.trend(alpha=0.24))


def assert_is_close(left, right):
    msg = f"{left} is not close to {right}."
    assert math.isclose(left, right), msg


if __name__ == "__main__":
    unittest.main()
