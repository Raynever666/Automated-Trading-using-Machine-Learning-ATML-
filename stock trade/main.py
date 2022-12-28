# NEED sklearn ver 1.0.2
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# this line disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(os.path.abspath(os.getcwd()))
import time
import requests as requests_cache
import sys
import datetime
import yfinance as yf
import joblib
from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest_async import gather_with_concurrency, AsyncRest
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from matplotlib import rc
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import pandas_datareader as web
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import numpy as np
from ast import Constant

# check gpu availability
print(tf.config.list_physical_devices('GPU'))

# depends on ur account
api_key_id = "YOUR-API-KEY"
api_secret = "YOUR-SECRET-KEY"
base_url = "https://paper-api.alpaca.markets"
feed = "iex"  # change to "sip" if you have a paid account

# fixed variables
time_steps = 5
symbol = ["AAPL", "AMD", "ASML", "JPM", "META",
          "MSFT", "NVDA", "PFE", "TSLA", "TSM"]
f_columns = ['open', 'high', 'low', 'volume', 'trade_count', 'vwap']

# initialize variables
# long and short list contain symbol and supposed quantity
long = []
short = []
longSymbol = []
shortSymbol = []
oldPosList = []
livePrice = [0] * len(symbol)
# in symbolRanking, the index refers to corresponding symbol
symbolPercent = [0] * len(symbol)
lastBalanceTime = float(0.0)
fourteenMinute = float(14 * 60)

# Wait for market to open.
def awaitMarketOpen():
    isOpen = api.get_clock().is_open
    while(not isOpen):
        clock = api.get_clock()
        openingTime = clock.next_open.replace(
            tzinfo=datetime.timezone.utc).timestamp()
        currTime = clock.timestamp.replace(
            tzinfo=datetime.timezone.utc).timestamp()
        timeToOpen = int((openingTime - currTime) / 60)
        print(str(timeToOpen) + " minutes til market open.")
        time.sleep(60)
        isOpen = api.get_clock().is_open

# Submit an order if quantity is above 0.
def submitOrder(stock: str, qty: int, side: str, respond=[]):
    if(qty > 0):
        try:
            api.submit_order(stock, qty, side, "market", "day")
            print("Market order of | " + str(qty) + " " +
                  stock + " " + side + " | completed.")
            respond.append(True)
        except:
            print("Order of | " + str(qty) + " " + stock +
                  " " + side + " | did not go through.")
            respond.append(False)
    else:
        print("Quantity is 0, order of | " + str(qty) +
              " " + stock + " " + side + " | completed.")
        respond.append(True)

# get live price from yahoo
def getLivePrice(symbol: str, session):
    while True:
        try:
            data = yf.download(tickers=symbol, period='10m', interval="1m",
                               progress=False, session=session, threads=False, timeout=2)
            return data.Close[len(data) - 1]
        except:
            time.sleep(1)

# predict percentage change
def predictPercentChange():
    session = requests_cache.Session()
    for i in range(len(symbol)):
        # get and save live price
        livePrice[i] = getLivePrice(symbol[i], session)

        # get past data
        pastData = api.get_bars(symbol[i], TimeFrame(
            15, TimeFrameUnit.Minute), adjustment='all').df

        # scale past input data
        pastData.loc[:, f_columns] = f_transformer[i].transform(
            pastData[f_columns].to_numpy())
        pastData['close'] = close_transformer[i].transform(pastData[['close']])

        # numpy to array
        past_X = []
        past_X.append(pastData.iloc[len(pastData) -time_steps - 1:len(pastData) - 1])
        past_X = np.array(past_X)

        # predict using past data
        future_Y = model[i].predict(past_X)

        # inverse scale of predicted price
        future_Y = close_transformer[i].inverse_transform(future_Y)
        future_Y = future_Y.flatten()

        # calculate percent change
        # symbolPercent is the predicted percent change
        symbolPercent[i] = float(
            ((future_Y - livePrice[i]) * 100) / livePrice[i])
        print(symbol[i] + " had predicted percent change: " +
              str(symbolPercent[i]))

        time.sleep(0.8)

    session.close()

# calculate position of stocks and the supposed amount of holding
def getPosition():
    # predict percentage change
    predictPercentChange()

    # put stock in long and short list and determine buying amount
    # reset long and short list
    long[:] = []
    short[:] = []
    totalPercentChange = sum(map(abs, symbolPercent))
    buyingPower = float(api.get_account().last_equity) * (1 + (totalPercentChange/100))
    for i in range(len(symbol)):
        # determine supposed amount in USD for holding
        amount = buyingPower * (abs(symbolPercent[i]) / totalPercentChange)

        # determine quantity of buying
        qty = int(amount / livePrice[i])

        # minimum quantity is 1 short/long
        if (qty == 0):
            qty = 1

        # if predicted rise, buy long
        if symbolPercent[i] > 0:
            long.append([symbol[i], qty])

        # else sell short as predicted fall
        else:
            short.append([symbol[i], qty])

# return index pos of symbol in short list
def shortListIndex(symbol: str):
    # find index in long list
    for i in range(len(short)):

        # if it is a match return
        if (short[i][0] == symbol):
            return i

    # if index cannot be found crash the program
    sys.exit()

# return index pos of symbol in long list
def longListIndex(symbol: str):
    # find index in long list
    for i in range(len(long)):

        # if it is a match return
        if (long[i][0] == symbol):
            return i

    # if index cannot be found crash the program
    sys.exit()

# supposed all position exist
def adjustPos():
    for oldPos in oldPosList:

        oldQty = abs(int(float(oldPos.qty)))

        # Position is now not in long list
        if(longSymbol.count(oldPos.symbol) == 0):

            # Position is now not in short list either.  Clear position.
            if (shortSymbol.count(oldPos.symbol) == 0):
                api.close_position(oldPos.symbol)

            # Position is now in short list
            else:

                # find wanted qty of particular stock in short list
                shortQty = short[shortListIndex(oldPos.symbol)][1]

                # position was in long but now in short. clear position and sell short
                if (oldPos.side == "long"):
                    api.close_position(oldPos.symbol)
                    time.sleep(2.2)
                    submitOrder(oldPos.symbol, shortQty, "sell")

                # position was in short and is now in short list
                else:

                    # old quantity is what what we want, pass for now
                    if (oldQty == shortQty):
                        pass

                    # need to adjust qty
                    else:
                        diff = oldQty - shortQty
                        # too much short, buy some back
                        if (diff > 0):
                            submitOrder(oldPos.symbol, abs(diff), "buy")
                        else:
                            submitOrder(oldPos.symbol, abs(diff), "sell")

        # position is now in long list
        else:

            # find wanted qty of particular stock in long list
            longQty = long[longListIndex(oldPos.symbol)][1]

            # position changed from short to long, clear old position and buy long
            if (oldPos.side == "short"):
                api.close_position(oldPos.symbol)
                time.sleep(2.2)
                submitOrder(oldPos.symbol, longQty, "buy")

            # position was in long and is now in long list
            else:

                # old quantity is what what we want, pass for now
                if (oldQty == longQty):
                    pass

                 # need to adjust qty
                else:
                    diff = oldQty - longQty
                    # too much long, sell some out
                    if (diff > 0):
                        submitOrder(oldPos.symbol, abs(diff), "sell")
                    else:
                        submitOrder(oldPos.symbol, abs(diff), "buy")

# supposed no existing position WIP
def takePos():

    # buy in all long
    for i in range(len(long)):
        submitOrder(long[i][0], long[i][1], "buy")

    # sell out all short
    for i in range(len(short)):
        submitOrder(short[i][0], short[i][1], "sell")

# rebalance position
def rebalance():

    # clear all orders
    api.cancel_all_orders()

    # get list of positions
    getPosition()

    # print symbol and qty of long position
    longSymbol[:] = []
    for i in range(len(long)):
        longSymbol.append(long[i][0])
    print("We are taking a long position in: " + str(long))
    # print symbol and qty of short position
    shortSymbol[:] = []
    for i in range(len(short)):
        shortSymbol.append(short[i][0])
    print("We are taking a short position in: " + str(short))

    global oldPosList
    oldPosList.clear()
    oldPosList = api.list_positions()
    # adjust position if oldPosList is not empty (i.e. position already exist)
    if len(oldPosList) > 0:
        print("adjusting position...")
        adjustPos()
        print("adjustment completed")

    # else submit orders according to list directly
    else:
        print("taking new position...")
        takePos()
        print("position submitted")

if __name__ == "__main__":

    # connect to alpaca_trade_api
    print("Connecting to alpaca_trade_api...")
    rest = AsyncRest(key_id=api_key_id,
                     secret_key=api_secret)

    api = tradeapi.REST(key_id=api_key_id,
                        secret_key=api_secret,
                        base_url=base_url)
    print("connected")

    # load tensorflow model
    model = []
    print("loading model...")
    for i in range(len(symbol)):
        model_fileName = symbol[i] + "_model"
        mod = tf.keras.models.load_model(model_fileName)
        model.append(mod)
        print("loaded model: ", model_fileName)
    print("model loaded successfully")

    # load the robust transformer
    # NEED sklearn ver 1.0.2
    f_transformer = []
    close_transformer = []
    print("loading transformer...")
    for i in range(len(symbol)):
        f_transformer_filename = "f_transformer_" + symbol[i] + ".save"
        close_transformer_filename = "close_transformer_" + symbol[i] + ".save"

        f_tran = joblib.load(f_transformer_filename)
        close_tran = joblib.load(close_transformer_filename)

        f_transformer.append(f_tran)
        close_transformer.append(close_tran)
    print("transformer loaded successfully")

    # initialize clock
    clock = api.get_clock()

    # First, cancel any existing orders so they don't impact our buying power.
    print("clearing existing orders")
    api.cancel_all_orders()
    print("orders have been cleared successfully")

    # infinite loop for everyday trade
    while True:
        # await market open when not in next daily loop
        print("Waiting for market to open...")
        awaitMarketOpen()
        print("Market opened.")

        # loop while market is open
        # until market is closed
        isOpen = True
        while isOpen:

            # update isOpen
            isOpen = api.get_clock().is_open

            # break if market had closed
            if not isOpen:
                print("market is not open")
                break

            # action if market is open
            else:
                # get local system time
                currTime = time.time()

                # update clock
                clock = api.get_clock()

                # Figure out when the market will close and current time
                server_closingTime = clock.next_close.replace(
                    tzinfo = datetime.timezone.utc).timestamp()
                server_currTime = clock.timestamp.replace(
                    tzinfo = datetime.timezone.utc).timestamp()

                # timeToClose in sec
                timeToClose = server_closingTime - server_currTime

                # clear all positions if 2 min left, else trade every 15 min
                if (timeToClose < (2 * 60)):
                    print("market closing in 2 minute, clearing all positions...")
                    api.close_all_positions()
                    print("positions closed, sleeping until the market is closed")
                    time.sleep(60 * 2)

                # rebalance positions if 14 min mark had reached
                # else sleep for 1 min
                elif (currTime - lastBalanceTime >= fourteenMinute):
                    rebalance()

                    # get lastBalanceTime from system and (newTime - oldTime)
                    lastBalanceTime = time.time()
                    correctionTime = lastBalanceTime - currTime

                    print("rebalancing used " + str(correctionTime) + " sec")
                    time.sleep(60 - correctionTime)

                # sleep for 60s
                else:
                    timeSinceBalance = currTime - lastBalanceTime
                    print(str(timeSinceBalance / 60) +
                          " minutes since last rebalance")

                    # newTime - oldTime
                    correctionTime = time.time() - currTime
                    time.sleep(60 - correctionTime)
