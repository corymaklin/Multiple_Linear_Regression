# import libraries
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# import dataset
dataset = pandas.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:,-1].values
y = y.reshape(-1,1)

# encoding categorical data
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])
X = OneHotEncoder(categorical_features = [3]).fit_transform(X).toarray()

# avoiding the dummy variable trap
X = X[:,1:]

# building the optimal model using backward elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = numpy.delete(x, j, 1)
    regressor_OLS.summary()
    return x

X = numpy.append(arr = numpy.ones((50, 1)).astype(int), values = X, axis = 1)
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

# fitting multiple linear regression model to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)

