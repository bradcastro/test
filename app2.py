import streamlit as st
import os
os.system('pip install ccxt==1.59.2')
import ccxt
import pandas as pd
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import talib.abstract as ta
import logging
import math

logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

api_key = os.environ.get('BdsqHhEOpeLf9axXtKoMI2jzNFODlqPjd4Wt1R7acHvOpOnL3gmzMRCN11XHnab0')
api_secret = os.environ.get('3cyxoje0kcgMiSlydg4tb4VJd7F5hOXrSa2T7sGnDAkmAsFE8Xln7LSaQRrcIPf5')
client = Client(api_key, api_secret)

trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

def calculate_sma(data, window):
    return data['close'].rolling(window=window).mean()

def get_historical_data(symbol, interval):
    try:
        bars = client.get_historical_klines(symbol, interval, "1 month ago UTC")
        data = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
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

def execute_trade(symbol, side, cost, order_type=Client.ORDER_TYPE_MARKET):
    try:
        order = {}
        if side == "buy":
            order = client.order_market_buy(symbol=symbol, quantity=cost)
        elif side == "sell":
            order = client.order_market_sell(symbol=symbol, quantity=cost)
        return order
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        raise e

def main():
    pair = st.selectbox('Select trading pair', trading_pairs)
    data = get_historical_data(pair, Client.KLINE_INTERVAL_1DAY)

    model = train_model(data)

    st.write(f"Model trained on {pair} data with accuracy {model.score}.")

    latest_data = get_historical_data(pair, Client.KLINE_INTERVAL_1DAY)
    latest_data['SMA'] = calculate_sma(latest_data, 14)
    X_latest = latest_data.drop('date', axis=1).dropna()
    
    prediction = model.predict(X_latest.values.reshape(1, -1))

    if prediction[0] == 1:
        st.write(f"Model prediction: BUY {pair}.")
        execute_trade(pair, "buy", 0.001)
    elif prediction[0] == 0:
        st.write(f"Model prediction: SELL {pair}.")
        execute_trade(pair, "sell", 0.001)

if __name__ == '__main__':
    main()
