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

import io
import itertools
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall
import streamlit as st

import trend


def main():
    st.title("Trend analysis")
    uploaded = st.file_uploader(label="Upload")
    if uploaded is None:
        st.stop()

    df = pd.read_csv(uploaded, index_col=0, usecols=[0, 1], parse_dates=True)
    data = df.groupby(df.index.year).mean().squeeze().rename_axis(index="Year")
    st.dataframe(data)

    show_figure(plot_data(data=data).figure)

    min_years = st.number_input(
        label="Minimum number of years", value=max(len(data), 11) - 10, min_value=0
    )

    st.dataframe(all_periods(data=data, min_years=min_years))


def plot_data(data):
    ax = data.plot(ylim=(0, None), grid=True, linewidth=2)
    t = trend.trend_test(x=data, kind="linear")
    plt.plot(
        data.index[[0, -1]],
        np.array([0, len(data)]) * t.slope + t.intercept,
        label="Linear",
    )
    t = trend.trend_test(
        x=data, kind="mann-kendall", mk_test=pymannkendall.original_test
    )
    ax.plot(
        data.index[[0, -1]],
        np.array([0, len(data)]) * t.slope + t.intercept,
        label="Mann-Kendall",
    )
    ax.legend()
    return ax


def all_periods(data, min_years):
    return pd.DataFrame(
        {
            "start": x.index[0],
            "end": x.index[-1],
            "num_years": len(x),
            "linear_trend": trend.trend_test(x=x, kind="linear").trend(),
            "mk_trend": trend.trend_test(
                x=x, kind="mann-kendall", mk_test=pymannkendall.original_test
            ).trend(),
        }
        for x in (data.iloc[s] for s in get_slices(x=data, min_years=min_years))
    )


def get_slices(x, min_years):
    def _get_slices(x, k):
        return (slice(i, i - k if i < k else None) for i in range(k + 1))

    return itertools.chain(*(_get_slices(x, k) for k in range(len(x) - min_years + 1)))


def show_figure(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    url = f"""<img src="data:image/svg+xml,{urllib.parse.quote(buf.getvalue())}">"""
    st.markdown(url, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
