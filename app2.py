import os
import pandas as pd
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import math
import ccxt
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import qtpylib.indicators as qtpylib


class SimpleMA_strategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in the documentation: https://www.freqtrade.io/en/latest/strategy-customization/
    You can:
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    """
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.1
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Add your indicators here
        # Example:
        # dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # Add your buy signal logic here
        # Example:
        # dataframe.loc[dataframe['rsi'] < 30, 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        # Add your sell signal logic here
        # Example:
        # dataframe.loc[dataframe['rsi'] > 70, 'sell'] = 1

        return dataframe


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
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume', 'ignore'])
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
        latest_data = get_historical_data(pair, Client.KLINE_INTERVAL_1HOUR, "1 hour ago UTC", "now UTC")
        if latest_data is None:
            continue

        # Make prediction
        X_latest = pd.concat([latest_data['close'].pct_change(), latest_data['volume'].pct_change()], axis=1).dropna()
        prediction = model.predict(X_latest.iloc[-1])

        # Execute trades based on prediction
        if prediction == 1:
            execute_trade(pair, Client.SIDE_BUY, 100)
        elif prediction == 0:
            execute_trade(pair, Client.SIDE_SELL, 100)


if __name__ == '__main__':
    main()
