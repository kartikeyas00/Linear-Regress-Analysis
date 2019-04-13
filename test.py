# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:34:38 2019

@author: karti
"""
import pandas as pd
from matplotlib import pyplot as plt

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
plt.savefig('Time for different Methods.png',bbox_inches='tight',dpi=400)
plt.show()
