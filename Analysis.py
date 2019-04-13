# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:58:39 2019

@author: ksharma
"""

import numpy as np
from scipy import stats,optimize
import pandas as pd
from LinRegress_Classic import LinRegress_Classic
from LinRegress_GD import LinRegress_GD
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate
import time


billion = 1000000000

gdp_india=pd.read_csv('India_GDP.csv')
gdp=np.array(gdp_india[(gdp_india['Year']>=1996)&(gdp_india['Year']<=2018)]['GDP($)'])/billion

forexp_india=pd.read_csv('India_Forexp.csv')
forexp=np.array(forexp_india['Total Export($)'])/billion

"""
Lets consider Foreign export be x and GDP be y
"""


def sklearn_LinearModel(x,y):
    regression=LinearRegression(fit_intercept=True)
    regression.fit(x,y)
    y_predict=regression.predict(x)
    plt.plot(x, y, 'o')
    plt.plot(x,y_predict,label='Regression Line', linestyle='--')
    plt.xlabel('Foreign Export(In Billion $)')
    plt.ylabel('GDP(In Billion $)')
    plt.legend(loc='best')
    plt.show()
    return [regression.intercept_,regression.coef_[0],'y = '+str(regression.coef_[0])+'x'+' + '+str(regression.intercept_),(regression.score(x,y))]


def LinRegress_Classic_LinearModel(x,y,xlabel,ylabel):
    regression=LinRegress_Classic(x,y,xlabel,ylabel)
    regression.get_equation()
    regression.plot_equation()
    return [regression.intercept(),regression.slope(),regression.get_equation(),regression.calc_determination_coeff()]


def LinRegress_GD_LinearModel(x,y,z,iterations):
    regression=LinRegress_GD(x,y,learning_rate=z,num_iterations=iterations)
    print(regression.get_equation())
    regression.plot_equation()
    return [regression.gradient_descent(x,y)[0],regression.gradient_descent(x,y)[1],regression.get_equation(),regression.calc_determination_coeff()]

def polyfit_LinearModel(x,y):
    regression=np.polyfit(forexp,gdp,deg=1)
    y_predict=regression[0]*x+regression[1]
    plt.plot(x, y, 'o')
    plt.plot(x,y_predict,label='Regression Line', linestyle='--')
    plt.xlabel('Foreign Export(In Billion $)')
    plt.ylabel('GDP(In Billion $)')
    plt.legend(loc='best')
    plt.show()
    return [regression[1],regression[0],'y = '+str(regression[0])+'x'+' + '+str(regression[1]),(regression[0]*(np.std(x)/np.std(y)))**2]

def linregress_LinearModel(x,y):
    regression=stats.linregress(x,y)
    y_predict=regression[0]*x+regression[1]
    plt.plot(x, y, 'o')
    plt.plot(x,y_predict,label='Regression Line', linestyle='--')
    plt.xlabel('Foreign Export(In Billion $)')
    plt.ylabel('GDP(In Billion $)')
    plt.legend(loc='best')
    plt.show()
    return [regression[1],regression[0],'y = '+str(regression[0])+'x'+' + '+str(regression[1]),regression[2]**2]


def curvefit_LinearModel(func,x,y):
    regression=optimize.curve_fit(func,forexp,gdp)[0]
    y_predict=regression[0]*x+regression[1]
    plt.plot(x, y, 'o')
    plt.plot(x,y_predict,label='Regression Line', linestyle='--')
    plt.xlabel('Foreign Export(In Billion $)')
    plt.ylabel('GDP(In Billion $)')
    plt.legend(loc='best')
    plt.show()
    return [regression[1],regression[0],'y = '+str(regression[0])+'x'+' + '+str(regression[1]),(regression[0]*(np.std(x)/np.std(y)))**2]

def func(x,a,b):
   return a*x+b

def ols_LinearModel(x,y):
    model=sm.OLS(y,sm.add_constant(x))
    regression=model.fit()
    y_predict=regression.predict()
    plt.plot(x, y, 'o')
    plt.plot(x,y_predict,label='Regression Line', linestyle='--')
    plt.xlabel('Foreign Export(In Billion $)')
    plt.ylabel('GDP(In Billion $)')
    plt.legend(loc='best')
    plt.show()
    return [regression.params[0],regression.params[1],'y = '+str(regression.params[1])+'x'+' + '+str(regression.params[0]),regression.rsquared]

start_time = time.time()
res_sklearn=sklearn_LinearModel(forexp.reshape(-1,1),gdp)
res_sklearnTime= (time.time() - start_time)

start_time = time.time()
res_LinRegClassic=LinRegress_Classic_LinearModel(forexp,gdp,'Foreign Export(In Billion $)','GDP(In Billion $)')
res_LinRegClassicTime=(time.time() - start_time)

start_time = time.time()
res_LinRegGD=LinRegress_GD_LinearModel(forexp,gdp,0.000001,100000000)
res_LinRegGDTime=(time.time() - start_time)

start_time = time.time()
res_Polyfit=polyfit_LinearModel(forexp,gdp)
res_PolyfitTime=(time.time() - start_time)

start_time = time.time()
res_Linregress=linregress_LinearModel(forexp,gdp)
res_LinregressTime= (time.time() - start_time)

start_time = time.time()
res_curevfit=curvefit_LinearModel(func,forexp,gdp)
res_curevfitTime=(time.time() - start_time)

start_time = time.time()
res_ols=ols_LinearModel(forexp,gdp)  
res_olsTime=(time.time() - start_time)

print(tabulate([[res_sklearnTime]+res_sklearn,[res_LinRegClassicTime]+res_LinRegClassic,[res_LinRegGDTime]+res_LinRegGD,[res_PolyfitTime]+res_Polyfit,[res_LinregressTime]+res_Linregress,[res_curevfitTime]+res_curevfit,[res_olsTime]+res_ols],headers=['Methods','Time To Run   
                     ','Intercept','Slope','Equation','R^2'],showindex=['sklearn.linear_model.LinearRegression','LinRegress_Classic','LinRegress_GD','numpy.polyfit','scipy.stats.linregress','scipy.optimize.curve_fit','Statsmodels.OLS'	], tablefmt='github'))