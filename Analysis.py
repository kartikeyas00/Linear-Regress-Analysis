# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:58:39 2019

@author: ksharma
"""

import numpy as np
from scipy import stats,optimize
import pandas as pd

gdp=pd.read_csv('India_GDP.csv',skiprows=4)
gdp_india=gdp[gdp['Country Name']=='India']
gdp_india.drop(gdp_india.columns[[1,2,3,-2,-1]], axis=1, inplace=True)
gdp_india=gdp_india.transpose().iloc[2:].reset_index().rename(index=str, columns={"index":"Year",107: "GDP($)"})