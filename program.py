# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Download Historical Data
# ---------------------------

# Set the ticker symbol and time period
ticker = 'AAPL'  # Apple Inc.
start_date = "2018-01-01"
end_date = "2023-01-01"

# Download the data from yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the data
print("Historical Data:")
print(data.head())

# 2. Calculate Moving Averages
# ---------------------------
short_window = 40  # e.g., 40-day moving average
long_window = 100  # e.g., 100-day moving average

# Calculate the moving averages using the closing price
data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

# 3. Generate Trading Signals
# ---------------------------
# Create a 'Signal' column: 1 when short MA is above long MA (buy signal), else 0.
data['Signal'] = 0
data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 1

# Generate 'Position' signals: the day-to-day difference in the Signal.
# A change from 0 to 1 indicates a buy signal, and 1 to 0 indicates a sell signal.
data['Position'] = data['Signal'].diff()

# Display rows where a buy or sell signal occurred
print("\nTrading Signals:")
print(data.loc[data['Position'] != 0, ['Close', 'Short_MA', 'Long_MA', 'Signal', 'Position']].head())

# 4. Backtesting the Strategy
# ---------------------------

# Calculate the daily returns of the stock
data['Daily_Return'] = data['Close'].pct_change()

# Assume that when in the market (Signal==1) you get the stock's return and when not you get 0.
# We use the previous day's signal (using .shift(1)) to avoid bias.
data['Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)

# Calculate cumulative returns for both the market and the strategy.
data['Cumulative_Market_Return'] = (1 + data['Daily_Return']).cumprod()
data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

# 5. Plot the Results
# ---------------------------

# Plot cumulative returns: market vs. strategy
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Cumulative_Market_Return'], label='Market Returns', color='blue')
plt.plot(data.index, data['Cumulative_Strategy_Return'], label='Strategy Returns', color='orange')
plt.title('Market Returns vs. Strategy Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Plot the stock's closing price along with the moving averages and mark buy/sell signals
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
plt.plot(data.index, data['Short_MA'], label=f'{short_window}-Day MA', linestyle='--')
plt.plot(data.index, data['Long_MA'], label=f'{long_window}-Day MA', linestyle='--')

# Plot buy signals: where Position == 1
plt.scatter(data.index[data['Position'] == 1],
            data['Close'][data['Position'] == 1],
            marker='^', color='green', s=100, label='Buy Signal')

# Plot sell signals: where Position == -1
plt.scatter(data.index[data['Position'] == -1],
            data['Close'][data['Position'] == -1],
            marker='v', color='red', s=100, label='Sell Signal')

plt.title(f'{ticker} Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
