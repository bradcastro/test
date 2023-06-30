import os
import pandas as pd
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import math

# Initialize logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

api_key = 'BdsqHhEOpeLf9axXtKoMI2jzNFODlqPjd4Wt1R7acHvOpOnL3gmzMRCN11XHnab0'
api_secret = '3cyxoje0kcgMiSlydg4tb4VJd7F5hOXrSa2T7sGnDAkmAsFE8Xln7LSaQRrcIPf5'
client = Client(api_key, api_secret)

# Define trading pairs
trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

# Fetch historical data
def get_historical_data(symbol, interval, start_time, end_time):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time)
        data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        data['close'] = pd.to_numeric(data['close'])
        return data
    except Exception as e:
        logging.error(f"Error getting historical data: {e}")
        return None

# Train a RandomForestClassifier
def train_model(data):
    try:
        y = data['close'].pct_change().apply(lambda x: 1 if x > 0 else 0)
        X = pd.concat([data['close'].pct_change(), data['volume'].pct_change()], axis=1).dropna()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

# Execute trades
def execute_trade(symbol, side, cost, order_type=Client.ORDER_TYPE_MARKET):
    try:
        info = client.get_symbol_info(symbol)
        step_size = 0.0
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])

        precision = int(round(-math.log(step_size, 10), 0))  
        quantity = float(cost / client.get_avg_price(symbol=symbol)['price'])
        quantity = round(quantity, precision)

        order = client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity)
        return order
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return False

# Main function
def main():
    for pair in trading_pairs:
        # Get historical data
        data = get_historical_data(pair, Client.KLINE_INTERVAL_1HOUR, "90 days ago UTC", "now UTC")
        if data is None:
            continue
        
        # Train the model
        model = train_model(data)
        if model is None:
            continue
        
        # Get latest data for decision making latest_data = get_historical_data(pair, Client.KLINE_INTERVAL_1HOUR, "1 hour ago UTC")
        if latest_data is None:
            continue
        
        # Make prediction
        X_latest = pd.concat([latest_data['close'].pct_change(), latest_data['volume'].pct_changeI apologize for the confusion earlier. Here's the corrected code without any incorrect formatting:

```python
import os
import pandas as pd
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import math

# Initialize logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

api_key = 'BdsqHhEOpeLf9axXtKoMI2jzNFODlqPjd4Wt1R7acHvOpOnL3gmzMRCN11XHnab0'
api_secret = '3cyxoje0kcgMiSlydg4tb4VJd7F5hOXrSa2T7sGnDAkmAsFE8Xln7LSaQRrcIPf5'
client = Client(api_key, api_secret)

# Define trading pairs
trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

# Fetch historical data
def get_historical_data(symbol, interval, start_time, end_time):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time)
        data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        data['close'] = pd.to_numeric(data['close'])
        return data
    except Exception as e:
        logging.error(f"Error getting historical data: {e}")
        return None

# Train a RandomForestClassifier
def train_model(data):
    try:
        y = data['close'].pct_change().apply(lambda x: 1 if x > 0 else 0)
        X = pd.concat([data['close'].pct_change(), data['volume'].pct_change()], axis=1).dropna()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

# Execute trades
def execute_trade(symbol, side, cost, order_type=Client.ORDER_TYPE_MARKET):
    try:
        info = client.get_symbol_info(symbol)
        step_size = 0.0
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])

        precision = int(round(-math.log(step_size, 10), 0))  
        quantity = float(cost / client.get_avg_price(symbol=symbol)['price'])
        quantity = round(quantity, precision)

        order = client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity)
        return order
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return False

# Main function
def main():
    for pair in trading_pairs:
        # Get historical data
        data = get_historical_data(pair, Client.KLINE_INTERVAL_1HOUR, "90 days ago UTC", "now UTC")
        if data is None:
            continue
        
        # Train the model
        model = train_model(data)
        if model is None:
            continue
        
        # Get latest data for decision making
        latest_data = get_historical_data(pair, Client.KLINE_INTERVAL_1HOUR, "1 hour ago UTC", "
