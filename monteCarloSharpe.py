import time
import random as rd
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

#This program optimizes given portfolio for sharpe ratio
print("This program optimizes a portfolio chosen by monte carlo in terms of sharpe ratio. \n\n")
print("Market Cap is over a billion for every stock chosen")
num_stocks = int(input("How many stocks do you want in the portofolio?"))

simulations = 100000    #This is num simulations that monte carlo will run. Change to desired num
all_stock_data = pd.read_csv("StockCsv/all_stock_data.csv")
#keep only #'s
all_stock_data = all_stock_data.select_dtypes(include=np.number)
#drop columns with any nans
all_stock_data = all_stock_data.dropna(axis=1,how='any')
#drop columns with constants
all_stock_data = all_stock_data.loc[:, all_stock_data.std() > 0]
default_weights = np.full(num_stocks, 1 / num_stocks)
#just using 0.037 but can use more sophisticated or link to external source if desired
rf = 0.037 
#this weight is good and probably doesn't need changed
simulation_list = []

def port_metrics(weights, returns): 
    mean_returns = returns.mean()*252
    exp_ret = mean_returns @ weights
    cov = returns.cov() * 252
    #dot product for std dev. just a formula basically
    std = np.sqrt(weights @ cov @ weights)
    #stopping div by 0
    sharpe = (exp_ret - rf) / std if std != 0 else -np.inf
    return exp_ret, std, sharpe

def min_sharpe(weights, returns):
    return -port_metrics(weights, returns)[2]
    #To maximize sharp ratio you have to minimize the negative

def check_sum(weights):
    return sum(weights) - 1

for _ in range(simulations):
    try:
        # Pick 6 random stocks
        cols = rd.sample(list(all_stock_data.columns), num_stocks)
        data = all_stock_data[cols].dropna(axis=1, how='all')
        #skip if less than 2 rows (cannot calculate returns)
        if data.shape[0] < 2:
            continue
        #calculate daily log returns
        returns = np.log(data).diff().dropna()
        if not np.isfinite(returns.values).all():
            continue
        #skip if all zeros or constant
        if returns.std().sum() == 0:
            continue
        exp_ret, std, sharpe = port_metrics(default_weights, returns)
        if np.isnan(sharpe):
            continue
        simulation_list.append([data, sharpe])
    except Exception as e:
        print(f"Skipped iteration due to error: {e}")
        continue

#sorting by sharpe
simulation_list.sort(key=lambda x: x[1], reverse=True)

#getting top 100 candidates from sorted list
weight_optimize_list = [row[0] for row in simulation_list[:100]]
optimized_sharpe_list = []

    
for i in weight_optimize_list:
    returns = np.log(i).diff().dropna()
    if not np.isfinite(returns.values).all():
        continue  # skip this iteration
    n_stocks = i.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    bounds = [(0.03, 0.33) for _ in range(num_stocks)]
    constraints = {"type": "eq", "fun": check_sum}

    #minimize negative Sharpe
    portfolio = minimize(min_sharpe, weights, args=(returns,), method="SLSQP", bounds=bounds, constraints=constraints)

    exp_ret, std, sharpe = port_metrics(portfolio.x, returns)
    #skip portfolios with NaN Sharpe
    if np.isnan(sharpe) or std == 0:
        continue
    optimized_sharpe_list.append([i, sharpe, pd.Series(portfolio.x, index=returns.columns)])

optimized_sharpe_list.sort(key=lambda x: x[1], reverse = True)

for row in optimized_sharpe_list:
    print(f"Sharpe: {row[-2:]}\n")

#gives you metrics about the portfolio e.g. exp ret, std, sharpe

#the best portfolios in the sim
print("\033c", end="")
print("This will display top 10 sharpe ratio portfolios now.\n")
for i in range(0,10):
    best_data, best_sharpe, best_weights = optimized_sharpe_list[i]
    best_returns = np.log(best_data).diff().dropna()
    best_exp_ret, best_std, best_sharpe = port_metrics(best_weights.values, best_returns)
    print(f"Portfolio: {i+1}")
    print(optimized_sharpe_list[i][2].sort_values(ascending=False))
    print(f"Expected Ret:\t{best_exp_ret:>10.2%}")
    print(f"Volatility:\t{best_std:>10.2%}")
    print(f"Sharpe:\t\t{best_sharpe:>.2%}\n\n")



'''
instantaneous rate of return is  d(ln(stock))/dt or P'(t) / P(t)
where P(t) is stock price
Going to have array of weights and returns that are "multiplied" together

Maximizing for Sharpe Ratio e.g. how much return do I get per unit of risk?
Sharpe: ExpRet(Portfolio) - risk-free / stdDev(portfolio)
'''
