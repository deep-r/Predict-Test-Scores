# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:42:10 2018

@author: Deepti R.

Multiple linear regression practice on 'psychology test' dataset with bkward elimination

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('Psychology test.xls')

X = data.iloc[:,0:3].values
y = data.iloc[:,3].values

"""scatter plot"""

plt.scatter(X[:,0],y, color = "red")
plt.scatter(X[:,1],y, color = "blue")
plt.scatter(X[:,2],y, color = "green")

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

"""=======================backward elimination=========================="""
""" OLS is ordinry least squares"""
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((25,1)).astype(int), values = X, axis = 1)

X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[1,2,3]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

"""YAY!"""