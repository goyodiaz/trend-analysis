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


from dataclasses import dataclass

import scipy.stats as ss


def trend_test(x, kind, mk_test=None):
    if kind == "linear":
        wrapped = ss.linregress(x=range(len(x)), y=x)
        result = TrendTestResult(
            slope=wrapped.slope,
            intercept=wrapped.intercept,
            pvalue=wrapped.pvalue,
        )
    elif kind == "mann-kendall":
        wrapped = mk_test(x)
        result = TrendTestResult(
            slope=wrapped.slope, intercept=wrapped.intercept, pvalue=wrapped.p
        )
    else:
        raise ValueError("Unknown kind.")
    result.wrapped = wrapped
    return result


@dataclass
class TrendTestResult:
    slope: float
    intercept: float
    pvalue: float

    def trend(self, alpha=0.05):
        return self.pvalue < alpha
