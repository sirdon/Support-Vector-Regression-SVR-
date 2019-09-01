import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
# Importing the dataset
dataset = pd.read_csv(r'C:\Users\infor_000\Desktop\Machine Learning A-ZGäó Hands-On Python & R In Data Science\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 6 - Polynomial Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#fitting svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
 

#predict a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualising the polynomial regression results
mpl.scatter(X, y, color='red')
mpl.plot(X, regressor.predict(X), color='blue')
mpl.xlabel('position')
mpl.ylabel('salary')
mpl.title('polynomial regression visualisation')
mpl.show()
