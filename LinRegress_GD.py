# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:01:29 2019

@author: karti
"""

import matplotlib.pyplot as plt
import numpy as np

"""
Linear regression line has an equation of the form Y=mX+b where Y is the 
dependent variable and X is the independent variable. 

Examples of Linear regressions can be:
    1) Predicting house prices in a city on the basis of the area of the house.
        In this case the area of the house is independent variable and the 
        housing price is a dependent variable.
    2) Predicting tax rate on the basis of the per capita income of a country,
        In this case the per capita income is the independent variable and the 
        tax rate is the dependent variable.
        
We will be talking about two different approaches to calculate the best fit line

1) Minimising the root mean squared deviation with the classical approach
2) Minimising the mean square error with gradient descent
 

What is gradient descent?
Gradient descent is the process of tweaking the linear regression parameter which
are m(slope) or b(intercept) until the mean square error is the lowest.

What is gradient?
Gradient is the change of the output of a fucntion when its parameter is changed. 
"""


class LinRegress_GD:
    
    def __init__(self,x,y,learning_rate=0.01,num_iterations=1000):
        self.x=np.array(x)
        self.y=np.array(y)
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
    
    def get_gradient_at_b(self,x, y, b, m):
      """
      We are finding the gradient of mean square error as the intercept changes:
          Basically:
              we find the sum of y_value - (m*x_value + b) for all the y_values 
              and x_values we have and then we multiply the sum by a factor of 
              -2/N. N is the number of points we have.
      """
      N = len(x)      
      b_gradient = -(2/N) *  sum(y - ((m * x) +b))  
      return b_gradient
    
    def get_gradient_at_m(self,x, y, b, m):
      """
        We are finding the gradient of mean square error as the slope changes:
          Basically:
              we find the sum of (x_value * y_value - (m*x_value + b)_ for all 
              the y_values and x_values we have and then we multiply the sum by 
              a factor of -2/N. N is the number of points we have.
      
      """
      N = len(x)
      m_gradient = -(2/N) * sum(x * (y - ((m * x) +b)))  
      return m_gradient
    
    def step_gradient(self,b_current, m_current, x, y, learning_rate):
        b_gradient = self.get_gradient_at_b(x, y, b_current, m_current)
        m_gradient = self.get_gradient_at_m(x, y, b_current, m_current)
        b = b_current - (learning_rate * b_gradient)
        m = m_current - (learning_rate * m_gradient)
        return [b, m]
      
    def gradient_descent(self,x, y, learning_rate, num_iterations):
      b = 0
      m = 0
      self.b_val=[]
      self.m_val=[]
      for i in range(num_iterations):
        b, m = self.step_gradient(b, m, x, y, learning_rate)
        self.b_val.append(b)
        self.m_val.append(m)
      return [b,m,self.b_val,self.m_val]

    def plot_iterationsVsParameters(self):
        plt.plot(list(range(1,self.num_iterations+1)),
                 self.gradient_descent(self.x,self.y,self.learning_rate,self.num_iterations)[2]
                 ,color='blue',label='Intercept')
        plt.plot(list(range(1,self.num_iterations+1)),
                 self.gradient_descent(self.x,self.y,self.learning_rate,self.num_iterations)[3],
                 self.m_val,color='orange',label='Slope')
        plt.legend(loc='best')
        plt.show()
        
    def calc_correlation_coeff(self):
        self.correlation_coeff= (self.gradient_descent(self.x,self.y,self.learning_rate,self.num_iterations)[1]*np.std(self.x))/np.std(self.y)
        return self.correlation_coeff
    
    def calc_determination_coeff(self):
        return np.square(self.correlation_coeff)
    
    def get_equation(self):
        self.b=self.gradient_descent(self.x,self.y,self.learning_rate,self.num_iterations)[0]
        self.m=self.gradient_descent(self.x,self.y,self.learning_rate,self.num_iterations)[1]
        self.equation='y = '+str(self.m)+' x + '+str(self.b)
        print(self.equation)
    
    def plot_equation(self):
        self.b=self.gradient_descent(self.x,self.y,self.learning_rate,self.num_iterations)[0]
        self.m=self.gradient_descent(self.x,self.y,self.learning_rate,self.num_iterations)[1]
        y_predict=(self.m * self.x ) + self.b #Y values based on the equation
        plt.plot(self.x,self.y,"o")
        plt.plot(self.x,y_predict,label=self.equation, linestyle='--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.show()
  
  
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]
Model1=LinRegress_GD(months,revenue)
Model1.get_equation()
print(Model1.calc_correlation_coeff())
print(Model1.calc_determination_coeff())
Model1.plot_equation()
