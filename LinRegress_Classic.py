# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:04:05 2019

@author: karti
"""

import matplotlib.pyplot as plt
import numpy as np

"""
We will be finding the line of best fit with the classical approach which is to minimize the root mean squared deviation, or equivalently, the sum of the squared deviations.

"""

class LinRegress_Classic:
    
     def __init__(self,x,y):
        self.x=np.array(x)
        self.y=np.array(y)
        self.x_mean=np.mean(self.x)
        self.y_mean=np.mean(self.y)
        self.x_std=np.std(self.x,ddof=1)
        self.y_std=np.std(self.y,ddof=1)
        
     def calc_correlation_coeff(self):
        N=len(self.x)-1
        self.correlation_coeff= sum(((self.x-self.x_mean)/self.x_std) * ((self.y-self.y_mean)/self.y_std))/N
        return self.correlation_coeff
    
     def calc_determination_coeff(self):
        return np.square(self.correlation_coeff)
    
     def slope(self):
         self.calc_correlation_coeff()
         self.m=self.correlation_coeff *(self.y_std/self.x_std)
         return self.m
     
     def intercept(self):
         self.slope()
         self.b=self.y_mean-(self.m*self.x_mean)
         return self.b
     
     def get_equation(self):
        self.b=self.intercept()
        self.m=self.slope()
        self.equation='y = '+str(self.m)+' x + '+str(self.b)
        print(self.equation)
        
     def plot_equation(self):
         self.b=self.intercept()
         self.m=self.slope()
         y_predict=(self.m * self.x ) + self.b
         plt.plot(self.x,self.y,"o")
         plt.plot(self.x,y_predict,label=self.equation, linestyle='--')
         plt.xlabel('x')
         plt.ylabel('y')
         plt.legend(loc='best')
         plt.show()
