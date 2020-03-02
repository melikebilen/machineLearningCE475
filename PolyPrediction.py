from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
X= df[['x1','x2','x3']][0:100].values
y= df['Y'][0:100].values
X_test_final=df[['x1','x2','x3']][100:]


poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

pre = pol_reg.predict(poly_reg.fit_transform(X_test_final))
print(pre)