# Support Vector Regression (SVR)

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', C=1.0, epsilon=0.1)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
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
plt.title(f'Support Vector Regression (MSR={MSR:.2f}, RMSE={RMSE:.2f}, MAE={MAE:.2f}, R2={r2:.2f})')
plt.show()