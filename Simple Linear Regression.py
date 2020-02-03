from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pnd
import numpy as np

dataFrame = pnd.read_csv("C:/Users/A780024/Desktop/ML World/ML practicals/Simple Linear Regression/SimpleLinearRegression.csv")

dataFrame.plot(x='Quantity',y='Price')
plt.show()

x = dataFrame['Quantity'].values.reshape(-1,1)
y = dataFrame['Price'].values.reshape(-1,1)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x, y)
    
# Predicting the price for Quantity 91
print(regressor.predict([[91]]))

#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)
