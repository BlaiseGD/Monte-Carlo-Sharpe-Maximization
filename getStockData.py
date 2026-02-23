import time
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

#This program optimizes given portfolio for sharpe ratio
print("This program optimizes a portfolio chosen by monte carlo (pre-screen) in terms of sharpe ratio. \n\n")
def get_data():
    nasdaq_df = pd.read_csv("StockCsv/NASDAQ.csv", usecols=['Symbol','MarketCap'])
    nyse_df = pd.read_csv("StockCsv/NYSE.csv",  usecols=['Symbol','MarketCap'])
    #combine the two and remove duplicates
    combined_df = pd.concat([nyse_df, nasdaq_df], ignore_index=True).drop_duplicates(subset="Symbol")
    #Filters based off market cap size (must be over a billion)
    filtered_df = combined_df[combined_df['MarketCap'].str.contains('B', na=False)]
    #gets stock data at close since specified date and drops the data that isn't available / doesn't work
    return filtered_df
#can change desired date (this is from date, goes to present)
date = "2024-01-01"
tickers = get_data()['Symbol'].tolist()
chunk_size = 75
all_data = []
for i in range(0,len(tickers), chunk_size):
    chunk = tickers[i:i+chunk_size]
    df = yf.download(chunk, date, threads=False)["Close"]
    all_data.append(df)
    time.sleep(1) #here to avoid rate limits

data = pd.concat(all_data, axis=1).dropna(axis=1, how='all')
#this creates the file with all stocks from NYSE and NASDAQ with mkt cap over a billion at time of data collected in those files.
#This is the file the main python file uses
data.to_csv("all_stock_data.csv")
