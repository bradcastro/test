import os
import ccxt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

api_key = 'BdsqHhEOpeLf9axXtKoMI2jzNFODlqPjd4Wt1R7acHvOpOnL3gmzMRCN11XHnab0'
api_secret = '3cyxoje0kcgMiSlydg4tb4VJd7F5hOXrSa2T7sGnDAkmAsFE8Xln7LSaQRrcIPf5'
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret
})

trading_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

def calculate_sma(data, window):
    return data['close'].rolling(window=window).mean()

def get_historical_data(symbol, interval):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, interval, since=exchange.milliseconds() - 30 * 24 * 60 * 60 * 1000)
        data = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['date'], unit='ms')
        data['close'] = pd.to_numeric(data['close'])
        return data
    except Exception as e:
        logging.error(f"Error getting historical data: {e}")
        raise e

def train_model(data):
    try:
        data['SMA'] = calculate_sma(data, 14)
        data = data.dropna()
        
        Y = (data['close'].shift(-1) > data['close']).astype(int)
        X = data.drop('date', axis=1)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
        model.fit(X_train, Y_train)
        
        logging.info(f"Model Score: {model.score(X_test, Y_test)}")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise e

def execute_trade(symbol, side, cost):
    try:
        order = {}
        if side == "buy":
            order = exchange.create_market_buy_order(symbol, cost)
        elif side == "sell":
            order = exchange.create_market_sell_order(symbol, cost)
        return order
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        raise e

def main():
    for pair in trading_pairs:
        data = get_historical_data(pair, '1d')

        model = train_model(data)

        latest_data = get_historical_data(pair, '1d')
        latest_data['SMA'] = calculate_sma(latest_data, 14)
        X_latest = latest_data.drop('date', axis=1).dropna()
        
        prediction = model.predict(X_latest.values.reshape(1, -1))

        if prediction[0] == 1:
            execute_trade(pair, "buy", 0.001)
        elif prediction[0] == 0:
            execute_trade(pair, "sell", 0.001)

if __name__ == '__main__':
    main()
