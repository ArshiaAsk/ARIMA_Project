import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt


np.random.seed(0)
plt.style.use('ggplot')
Tricker = 'GC=F'
Interval ='1d'
Date1 = dt.datetime(2005, 1, 1)
Date2 = dt.datetime(2025, 1, 1)

df = yf.download(tickers=Tricker, interval=Interval, start=Date1, end=Date2)
df.to_csv('World_Gold_price.csv')