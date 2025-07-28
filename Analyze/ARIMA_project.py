import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf



data = pd.read_csv(r"C:\Users\2023\Desktop\Arshia_project\Sanji_Project\DataFrame\Modify_World_Oil_price.csv")
print(data)



print(f"data info{data.info()}")
print(f"data describe{data.describe()}")
print(f"missing values {data.isnull().sum()}")



#set_datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)



#first_plot
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title('Oil Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()



#mplfinance_plot
mpf.plot(data.loc['2024/1/1':'2025/1/1'], type='candle', volume=True)
plt.show()



#remove_outliers
x = data[(data['Low'] < 10)].index
data.drop(x, inplace=True)
print(data)



#modifyed_plot
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title('Oil Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()



#set_new_data
new_data = data.loc['2024/1/1':'2025/1/1']



from statsmodels.tsa.stattools import adfuller

results = adfuller(new_data['Close'])

print('ADF Statistics:', results[0])
print('ADF p-value:', results[1])
print('Critical Values:')
for key, value in results[4].items():
    print(f" {key}: {value}")

if results[1] <= 0.05:
    print('reject H0')
else:
    print('accept H0 (non-stationary)')

new_data['Price_diff'] = new_data['Close'].diff()
new_data = new_data.dropna(subset=['Price_diff'])
result_diff = adfuller(new_data['Price_diff'])

print("ADF Statistic (Diff):", result_diff[0])
print("p-value (Diff):", result_diff[1])
print("Critical Values (Diff):")
for key, value in result_diff[4].items():
    print(f" {key}: {value}")

if result_diff[1] <= 0.05:
    print('reject H0')
else:
    print('accept H0 (non-stationary)')



print(new_data)



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(new_data['Price_diff'].dropna(), lags=40, ax=plt.gca())
plt.title('ACF')

plt.subplot(1, 2, 2)
plot_pacf(new_data['Price_diff'].dropna(), lags=40, ax=plt.gca())
plt.title('PACF')

plt.tight_layout()
plt.show()

import itertools
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

p = range(0, 3)
d = [1]
q = range(0, 3)
pdq_combinations = list(itertools.product(p, d, q))

results = []

for param in pdq_combinations:
    try:
        model = sm.tsa.ARIMA(new_data['Close'], order=param)
        result = model.fit()
        results.append((param, result.aic, result.bic))
        print(f'ARIMA{param} - AIC: {result.aic:.2f} - BIC: {result.bic:.2f}')
    except:
        continue

best_model = sorted(results, key=lambda x: x[1])[0]
print(f"best model: \n ARIMA{best_model[0]} - AIC: {best_model[1]:.2f}, BIC: {best_model[2]:.2f}")



train_size = int(len(new_data['Close']) * 0.8)
train, test = new_data.iloc[:train_size], new_data.iloc[train_size:]

best_pdq = best_model[0]
model = sm.tsa.ARIMA(train['Close'], order=best_pdq)
result = model.fit()

print(result.summary())



forecast = result.forecast(steps=len(test))
print(forecast)



plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Close'], label='Train', color='blue')
plt.plot(test.index, test['Close'], label='Test', color='red')
plt.plot(test.index, forecast, label='Forecast', color='green')

# plt.plot(pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast,              label='Forecast Price', color='red', linestyle='dashed')
plt.legend()
plt.title('Future Price Prediction with ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



from statsmodels.tsa.statespace.sarimax import SARIMAX

model_2 = SARIMAX(new_data['Close'], order=(2,1,2), seasonal_order=(2,1,2,30))
result_2 = model_2.fit()
print(result_2.summary())



forecast_2 = result_2.forecast(steps=len(test))
print(forecast_2)



plt.figure(figsize=(14, 7))
plt.plot(train.index, train['Close'], label='Train', color='blue')
plt.plot(test.index, test['Close'], label='Test', color='red')
plt.plot(test.index, forecast_2, label='Forecast', color='green')
plt.legend()
plt.title('Future Price Prediction with SARIMAX')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
plt.show()





