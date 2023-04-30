# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset = pd.read_csv(url, names=column_names,na_values="?", skipinitialspace = True, sep=" ", comment ="\t")
print(dataset.head())

# Data pre-processing
dataset.drop_duplicates(inplace = True)
dataset.fillna(dataset.mean(), inplace=True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance

# Mean Squared Error
from sklearn.metrics import mean_squared_error
MSR = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ",MSR)

# Root Mean Squared Error
RMSE = np.sqrt(MSR) 
print("Root Mean Squared Error: ",RMSE)

# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ",MAE)

# R-Squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-Squared: ",r2)

# Visualize the residuals
plt.scatter(y_test, y_pred)
plt.plot([0, 1], [0, 1], '--k')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'Random Forest Regression (MSR={MSR:.2f}, RMSE={RMSE:.2f}, MAE={MAE:.2f}, R2={r2:.2f})')
plt.show()