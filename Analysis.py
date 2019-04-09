# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:58:39 2019

@author: ksharma
"""

import numpy as np
from scipy import stats,optimize
import pandas as pd
import LinRegress_Classic
import LinRegress_GD
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

gdp_india=pd.read_csv('India_GDP.csv')
gdp=np.array(gdp_india[(gdp_india['Year']>=1996)&(gdp_india['Year']<=2018)]['GDP($)'])

forexp_india=pd.read_csv('India_Forexp.csv')
forexp=np.array(forexp_india['Total Export($)']).reshape(-1,1)

"""
Lets consider Foreign export be x and GDP be y
"""


def sklearn_LinearModel(x,y):
    regression=LinearRegression(fit_intercept=True)
    regression.fit(x,y)
    y_predict=regression.predict(x)
    plt.plot(x, y, 'o')
    plt.plot(x,y_predict,label='Regression Line', linestyle='--')
    plt.xlabel('Foreign Export($)')
    plt.ylabel('GDP($)')
    plt.legend(loc='best')
    plt.show()
    return [regression.intercept_,regression.coef_[0],'y = '+str(regression.coef_[0])+'x'+' + '+str(regression.intercept_),(regression.score(x,y))]

