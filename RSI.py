# Use of the relative Strength Index (RSI) and python to determine if a stock is being over bought or over sold
# If a stock is over sold is good time to buy it, if the stock is over bought is  good time to sell the stock
# The common time period to use for RSI is 14 days and it returns values in the scale of 0-100 w/ high and
# low level values marked as 70/30 80/20 90/10. Higher the high level and lower the low level indicates a stronger price moment on shift
# So f.e RSI is considered over bought above 70 and considered over sold when below 30

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
plt.style.use('fivethirtyeight')

start = dt.datetime(2019,8,31)
end = dt.datetime(2020,8,31)
MSFT = web.DataReader('MSFT', 'yahoo', start, end).reset_index()

#Set date as the index for the data
MSFT = MSFT.set_index(pd.DatetimeIndex(MSFT['Date'].values))

plt.figure(figsize=(12.2,4.5))
plt.plot(MSFT.index, MSFT['Adj Close'], label='Adj Close Price')
plt.title('Adj Close Price Hist')
plt.xlabel('Jan 2019 - Aug 2020', fontsize = 18)
plt.ylabel('Adj Close Price',fontsize = 18)
plt.show()

# Prepare data do calculate RSI
# Get difference in price from the precious day
delta = MSFT['Adj Close'].diff(1)

#Delete NaN
delta = delta.dropna()
#Get positive gains (up) and the negative gains (down)
up = delta.copy()
down = delta.copy()

up[up<0] = 0
down[down>0] = 0

#Get the time period (14 days)
period = 14
#Calculate Avg gain and Avg loss
avg_gain=up.rolling(window = period).mean()
avg_loss= abs(down.rolling(window = period).mean())

# RSI Calculation

# Calculate the Relative Strength (RS)
RS = avg_gain / avg_loss

# Calculate Relative S. Index (RSI)
RSI = 100.0 - (100.0 / (1.0 + RS))

plt.figure(figsize=(12.2, 4.5))
RSI.plot()
plt.show()

#Create new df
df = pd.DataFrame()
df['Adj Close Price'] = MSFT['Adj Close']
df['RSI'] = RSI

#Plot adj
plt.figure(figsize=(12.2, 4.5))
plt.plot(df.index, df['Adj Close Price'])
plt.title('Adj Close Price Hist')
plt.legend(df.columns.values, loc = 'upper left')
plt.show()

#plot corresponding RSI and significant levels
plt.figure(figsize=(12.2, 4.5))
plt.title('RSI')
plt.plot(df.index,df['RSI'])
plt.axhline(0, linestyle ='--', alpha = 0.5, color='gray')
plt.axhline(10, linestyle ='--', alpha = 0.5, color='orange')
plt.axhline(20, linestyle ='--', alpha = 0.5, color='green')
plt.axhline(30, linestyle ='--', alpha = 0.5, color='red')
plt.axhline(70, linestyle ='--', alpha = 0.5, color='red')
plt.axhline(80, linestyle ='--', alpha = 0.5, color='green')
plt.axhline(90, linestyle ='--', alpha = 0.5, color='orange')
plt.axhline(100, linestyle ='--', alpha = 0.5, color='gray')
plt.show()