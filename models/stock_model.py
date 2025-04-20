import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import scipy.stats as stats
import matplotlib.pyplot as plt
torch.set_num_threads(12)

df = pd.read_csv('stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"Raw data shape: {df.shape}, Date range: {df['Date'].min()} to {df['Date'].max()}")

train_df = df[df['Date'] <= '2023-12-31']
test_df = df[df['Date'] > '2023-12-31']

train_prices = train_df.pivot(index='Date', columns='Ticker', values='Close').ffill().bfill()
test_prices = test_df.pivot(index='Date', columns='Ticker', values='Close').ffill().bfill()

print(f"Train prices shape: {train_prices.shape}, NaNs: {train_prices.isna().sum().sum()}")
print(f"Test prices shape: {test_prices.shape}, NaNs: {test_prices.isna().sum().sum()}")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

features = []
for ticker in train_prices.columns:
    stock_data = df[df['Ticker'] == ticker].set_index('Date')
    stock_data['Returns'] = stock_data['Close'].pct_change().fillna(0)
    stock_data['Lag1_Return'] = stock_data['Returns'].shift(1).fillna(0)
    stock_data['Lag5_Return'] = stock_data['Returns'].shift(5).fillna(0)
    stock_data['Volatility'] = stock_data['Returns'].rolling(45, min_periods=1).std() * np.sqrt(252)
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    stock_data['Momentum'] = (stock_data['Close'] / stock_data['Close'].shift(45).replace(0, np.nan)).fillna(1)
    stock_data = stock_data[['Returns', 'Lag1_Return', 'Lag5_Return', 'Volatility', 'RSI', 'Momentum']].replace([np.inf, -np.inf], 0).fillna(0)
    stock_data['Ticker'] = ticker
    features.append(stock_data.reset_index())

feature_df = pd.concat(features)
feature_pivot = feature_df.pivot(index='Date', columns='Ticker')
feature_pivot.columns = [f"{col[0]}_{col[1]}" for col in feature_pivot.columns]
feature_pivot = feature_pivot.ffill().bfill()

print(f"Feature pivot shape: {feature_pivot.shape}, NaNs: {feature_pivot.isna().sum().sum()}")

scaler = MinMaxScaler()

train_features = feature_pivot[feature_pivot.index <= '2023-12-31']
test_features = feature_pivot[feature_pivot.index > '2023-12-31']

train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

print(f"Train scaled shape: {train_scaled.shape}, NaNs: {np.isnan(train_scaled).sum()}")
print(f"Test scaled shape: {test_scaled.shape}, NaNs: {np.isnan(test_scaled).sum()}")

def create_sequences(data, seq_length=45):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, :len(train_prices.columns)])  # Predict Returns for all tickers
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
X_test, y_test = create_sequences(test_scaled)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(45, train_scaled.shape[1])),
    Dropout(0.3),
    LSTM(256),
    Dropout(0.3),
    Dense(len(train_prices.columns))  # Predict Returns for 103 tickers
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Model Training Metrics')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np

# Get predictions from the model
test_pred = model.predict(X_test, verbose=0)

# Stack predictions with zeros to match scaler input shape
pred_zeros = np.zeros((test_pred.shape[0], train_scaled.shape[1] - len(train_prices.columns)))
pred_stacked = np.hstack([test_pred, pred_zeros])
test_pred_returns = scaler.inverse_transform(pred_stacked)

# Stack actual values with zeros
actual_zeros = np.zeros((y_test.shape[0], train_scaled.shape[1] - len(train_prices.columns)))
actual_stacked = np.hstack([y_test, actual_zeros])
test_actual_returns = scaler.inverse_transform(actual_stacked)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_actual_returns.flatten(), test_pred_returns.flatten()))
dir_acc = np.mean(np.sign(test_actual_returns.flatten()) == np.sign(test_pred_returns.flatten())) * 100

# Print results
print(f"Test RMSE: {rmse:.4f}, Test Directional Accuracy: {dir_acc:.2f}%")

## April 2025 Forecast with CSV Export
import numpy as np
import pandas as pd

# Last 45 days up to Mar 21, 2025
last_idx = feature_pivot.index <= '2025-03-21'
last_45_scaled = scaler.transform(feature_pivot[last_idx].tail(45))
print(f"Last 45 days shape: {last_45_scaled.shape}")  # (45, 618)

# Forecast 21 days (April 1-30, approx. trading days)
forecast = []
current_sequence = last_45_scaled.copy()
for _ in range(21):
    X_apr = current_sequence.reshape(1, 45, train_scaled.shape[1])
    pred = model.predict(X_apr, verbose=0)
    pred_zeros = np.zeros((pred.shape[0], train_scaled.shape[1] - len(train_prices.columns)))
    pred_stacked = np.hstack([pred, pred_zeros])
    pred_full = scaler.inverse_transform(pred_stacked)[:, :len(train_prices.columns)]
    forecast.append(pred_full[0])
    current_sequence = np.roll(current_sequence, -1, axis=0)
    current_sequence[-1, :len(train_prices.columns)] = pred_full[0]
    current_sequence[-1, len(train_prices.columns):] = current_sequence[-2, len(train_prices.columns):]

forecast_returns = np.array(forecast)
print(f"Apr forecast returns shape: {forecast_returns.shape}, mean: {forecast_returns.mean():.4f}")

# Portfolio metrics
trading_days = 21
metrics = pd.DataFrame({
    'Forecast_Return': forecast_returns.mean(axis=0) * trading_days * 100,
    'Volatility': forecast_returns.std(axis=0) * np.sqrt(252) * 100,
}, index=train_prices.columns)
metrics['Sharpe'] = (metrics['Forecast_Return'] / 100 * 252 - 0.02) / (metrics['Volatility'] / 100)

# Custom weights
def calculate_weights(portfolio):
    return_score = (portfolio['Forecast_Return'] - portfolio['Forecast_Return'].min()) / (portfolio['Forecast_Return'].max() - portfolio['Forecast_Return'].min() + 1e-6)
    vol_score = 1 - (portfolio['Volatility'] - portfolio['Volatility'].min()) / (portfolio['Volatility'].max() - portfolio['Volatility'].min() + 1e-6)
    weights = (return_score + vol_score) / (return_score + vol_score).sum()
    return weights

# Corrected portfolio definitions
portfolios = {
    'Minimum_Risk': metrics.sort_values('Volatility').iloc[:12],  # Lowest volatility
    'Moderate_Risk': metrics.sort_values('Sharpe', ascending=False).iloc[:12],  # Balanced risk/reward
    'High_Risk': metrics.sort_values('Forecast_Return', ascending=False).iloc[:12]  # Highest return, high vol
}

# Prepare CSV data
csv_data = []

for name, portfolio in portfolios.items():
    stocks = portfolio.index.tolist()
    weights = calculate_weights(portfolio)
    
    stock_indices = [train_prices.columns.get_loc(stock) for stock in stocks]
    apr_pred_port = forecast_returns[:, stock_indices].dot(weights)

    forecast_return = apr_pred_port.mean() * trading_days * 100
    volatility = apr_pred_port.std() * np.sqrt(252) * 100
    sharpe = (forecast_return / 100 * 252 - 0.02) / volatility
    var = -1.645 * apr_pred_port.std() * trading_days * 100

    # Weights as percentages
    weights_pct = [f"{w*100:.1f}%" for w in weights]
    
    # Print output
    print(f"\nApr_2025:")
    print(f"Portfolio: {name}")
    print(f"Stocks: {','.join(stocks)}")
    print(f"Weights: {','.join(weights_pct)}")
    print(f"Total Weight: {sum(weights)*100:.1f}%")
    print(f"Forecast: {forecast_return:.2f}% | Volatility: {volatility:.2f}% | Sharpe: {sharpe:.2f} | VaR: {var:.0f}")
    
    # Add to CSV data
    csv_data.append({
        'Portfolio': name,
        'Stocks': ','.join(stocks),
        'Weights': ','.join(weights_pct),
        'Total_Weight': f"{sum(weights)*100:.1f}%",
        'Forecast_Return': f"{forecast_return:.2f}%",
        'Volatility': f"{volatility:.2f}%",
        'Sharpe': f"{sharpe:.2f}",
        'VaR': f"{var:.0f}"
    })

# Save to CSV
df_csv = pd.DataFrame(csv_data)
df_csv.to_csv('Stock_LSTM_Forcast.csv', index=False)
print("\nSaved to 'Stock_LSTM_Forcas.csv'")
