import numpy as np

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


X = np.random.random((1000, 10))
a = np.random.random((10, 3))
y = np.dot(X, a) + np.random.normal(0, 1e-3, (1000, 3))

# fitting
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear')).fit(X, y)

# predicting
print(np.mean((multioutputregressor.predict(X) - y)**2, axis=0))  # 0.004, 0.003, 0.005












