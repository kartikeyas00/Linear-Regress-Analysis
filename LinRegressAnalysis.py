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

billion = 1e9

gdp_india=pd.read_csv('Data/India_GDP.csv')
gdp=np.array(gdp_india[(gdp_india['Year']>=1996)&(gdp_india['Year']<=2018)]['GDP($)'])/billion

forexp_india=pd.read_csv('Data/India_Forexp.csv')
forexp=np.array(forexp_india['Total Export($)'])/billion

"""
We are considering foreign export as the independent variable and the GDP as 
dependent variable. Hence x variable will be the foreign export and y will be
the GDP.
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
    plt.plot(x,y_predict,label='y = '+str(round(regression.params[1],2))+'x'+' + '+str(round(regression.params[0],2)), linestyle='--')
    plt.xlabel('Foreign Export(In Billion $)')
    plt.ylabel('GDP(In Billion $)')
    plt.legend(loc='best')
    plt.show()
    return [regression.params[0],regression.params[1],'y = '+str(regression.params[1])+'x'+' + '+str(regression.params[0]),regression.rsquared]

###############################################################################
    #Calling the above defined functions in the following order and also getting
    #the running time.
    #1 --> sklearn_LinearModel based on scikit-learn
    #2 --> LinRegress_Classic_LinearModel 
    #3 --> LinRegress_GD_LinearModel
    #4 --> polyfit_LinearModel
    #5 --> linregress_LinearModel
    #6 --> curvefit_LinearModel
    #7 --> ols_LinearModel
#1
start_time = time.time()
res_sklearn=sklearn_LinearModel(forexp.reshape(-1,1),gdp)
res_sklearnTime= (time.time() - start_time)

#2
start_time = time.time()
res_LinRegClassic=LinRegress_Classic_LinearModel(forexp,gdp,'Foreign Export(In Billion $)','GDP(In Billion $)')
res_LinRegClassicTime=(time.time() - start_time)

#3
start_time = time.time()
res_LinRegGD=LinRegress_GD_LinearModel(forexp,gdp,0.000001,100000000)
res_LinRegGDTime=(time.time() - start_time)

#4
start_time = time.time()
res_Polyfit=polyfit_LinearModel(forexp,gdp)
res_PolyfitTime=(time.time() - start_time)

#5
start_time = time.time()
res_Linregress=linregress_LinearModel(forexp,gdp)
res_LinregressTime= (time.time() - start_time)

#6
start_time = time.time()
res_curevfit=curvefit_LinearModel(func,forexp,gdp)
res_curevfitTime=(time.time() - start_time)

#7
start_time = time.time()
res_ols=ols_LinearModel(forexp,gdp)  
res_olsTime=(time.time() - start_time)

print(tabulate([[res_sklearnTime]+res_sklearn,[res_LinRegClassicTime]+res_LinRegClassic,[res_LinRegGDTime]+res_LinRegGD,[res_PolyfitTime]+res_Polyfit,[res_LinregressTime]+res_Linregress,[res_curevfitTime]+res_curevfit,[res_olsTime]+res_ols],headers=['Methods','Time To Run'   
                     ,'Intercept','Slope','Equation','R^2'],showindex=['sklearn.linear_model.LinearRegression','LinRegress_Classic','LinRegress_GD','numpy.polyfit','scipy.stats.linregress','scipy.optimize.curve_fit','Statsmodels.OLS'	], tablefmt='github'))
    
###############################################################################
    #Bar chart for the Time taken by the Linear regression functions above

    
df=pd.DataFrame({'Methods':['sklearn.linear_model','LinRegress_Classic','LinRegress_GD','numpy.polyfit','scipy.stats.linregress','scipy.optimize.curve_fit','Statsmodels.OLS'],
                 'Time':[res_sklearnTime,res_LinRegClassicTime,res_LinRegGDTime,res_PolyfitTime,res_LinregressTime,res_curevfitTime,res_olsTime]})

f, axis = plt.subplots(2, 1, sharex=True)
df.plot.bar(x='Methods',y='Time',ax=axis[0],legend=None,fontsize=15)
df.plot.bar(x='Methods',y='Time',ax=axis[1],legend=None,fontsize=15)
axis[0].set_ylim(10000, 16000)
plt.xlabel('Methods',fontsize=20)
plt.ylabel('Time (Seconds)',fontsize=20)
axis[1].set_ylim(0.01, 1)
axis[1].legend().set_visible(False)
axis[0].spines['bottom'].set_visible(False)
axis[1].spines['top'].set_visible(False)
axis[0].xaxis.tick_top()
axis[0].tick_params(labeltop='off')
axis[1].xaxis.tick_bottom()
d = .015
kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
axis[0].plot((-d,+d),(-d,+d), **kwargs)
axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
kwargs.update(transform=axis[1].transAxes)
axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
plt.savefig('Time for different Methods.png',bbox_inches='tight',dpi=600)
plt.show()    