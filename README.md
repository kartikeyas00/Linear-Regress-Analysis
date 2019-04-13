# Analysis of different methods and ways of Linear Regression in Python.

## Introduction
Linear regression is finding a line which best fits a set of data. Here we will be talking about the linear regression with only two variables, independent and the response variable.  
With two variables the typical equation of the regrssion line, which is also called the regression equation looks like  **Y=mX+b** where Y is the dependent variable and X is the independent variable.
 
 ***Examples of Linear regressions -***
 1. Predicting house prices in a city on the basis of the area of the house.
        In this case the area of the house is independent variable and the 
        housing price is a dependent variable.
 2. Predicting tax rate on the basis of the per capita income of a country,
        In this case the per capita income is the independent variable and the 
        tax rate is the dependent variable.
        
Here we will be displaying the variations in results and running time of the two Linear Regression methods written by me in Python and the builtin ways of doing Linear Regression in various libraries of Python.

## Linear Regression Methods (Python)

***Linear Regression Class written by me:***

| Class Name | Description |
| -----------| ------------|
| LinRegress_GD | Calculates the best fit model with minimising the mean square error with gradient descent |
| LinRegress_Classic | Calculates the best fit model by minimizing the root mean square deviation |

***Linear regression Methods from various Python libraries:***

| Method Name | Description |
| ------------| ------------|
| numpy.polyfit| This function fits a polynomial of a given degree to a set of Data by employing Least squares method.|
| scipy.stats.linregress | Highly specialized least squares regression which is only restricted to two sets of measurements.|
| scipy.optimize.curve_fit| Similar to "numpy.polyfit" but can fit any user-defined function to a data set by emplying the Least squares method|
| Statsmodels.OLS | Provides full blown statisitical information about the estimation process by employing Least squares method |
| sklearn.linear_model.LinearRegression | This is a classic method used by data scientist or in machine learning which is implemented with "scipy.linalg.lstsq"|

## Data
I would be performing Linear Regression on the Foreign Export and Gross Domestic Product Yearly data of India between 1996 and 2017. Here Foreign Export would be the Independent Variable and the Gross Domestic Product would be dependent on the Foreign Export. Also Foreign export is denoted as x and Gross Domestic Product is denoted as y. You can find the data in the folder *Data*.

## Results:

***Linear Model (From Statsmodels.OLS)***
![Screenshot](https://github.com/kartikeyas00/Linear-Regress-Analysis/blob/master/Plots/Linear%20Model%20(OLS).png)

***Results after performing Linear regression with the Python methods mentioned above***

| Methods                               |   Time To Run |   Intercept |   Slope | Equation                                    |      R^2 |
|---------------------------------------|---------------|-------------|---------|---------------------------------------------|----------|
| sklearn.linear_model.LinearRegression |     0.127449  |     190.277 | 6.23224 | y = 6.6.23224x + 190.277  | 0.931941 |
| LinRegress_Classic                    |     0.148438  |     190.277 | 6.23224 | y = 6.6.23224x + 190.277 | 0.931941 |
| LinRegress_GD                         | 15650.1       |     190.277 | 6.23224 | y = 6.6.23224x + 190.277   | 0.931941 |
| numpy.polyfit                         |     0.486575  |     190.277 | 6.23224 | y = 6.6.23224x + 190.277  | 0.931941 |
| scipy.stats.linregress                |     0.81231   |     190.277 | 6.23224 | y = 6.6.23224x + 190.277  | 0.931941 |
| scipy.optimize.curve_fit              |     0.109349  |     190.277 | 6.23224 | y = 6.6.23224x + 190.277 | 0.931941 |
| Statsmodels.OLS                       |     0.0949917 |     190.277 | 6.23224 | y = 6.6.23224x + 190.277 | 0.931941 |


***Below is the plot for all the methods' run time:***
![Screenshot](https://github.com/kartikeyas00/Linear-Regress-Analysis/blob/master/Plots/Time%20for%20different%20Methods.png)


