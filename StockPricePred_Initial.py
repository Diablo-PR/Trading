import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('bmh')

start = dt.datetime(2019,1,1)
end = dt.datetime(2020,8,31)
AAPL = web.DataReader('AAPL', 'yahoo', start, end).reset_index()

plt.figure(figsize=(16,8))
plt.title('APPLE')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(AAPL['Close'])
#plt.show()

df = AAPL[['Close']]
future_days = 25
df['Prediction'] = df[['Close']].shift(-future_days)

X = np.array(df.drop(['Prediction'],1))[:-future_days]
print(X)

y = np.array(df['Prediction'])[:-future_days]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tree=DecisionTreeRegressor().fit(x_train, y_train)
lr=LinearRegression().fit(x_train, y_train)

x_future = df.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

tree_prediction=tree.predict(x_future)
#print(tree_prediction)

lr_prediction=lr.predict(x_future)
#print(lr_prediction)

predictions = tree_prediction

valid = df[X.shape[0]:]
valid['Predictions']=predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()

predictions = lr_prediction

valid = df[X.shape[0]:]
valid['Predictions']=predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()