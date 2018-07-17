import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


def loadDataset(symbol):
    '''Loads Dataset of the passed symbol from the datasets folder.'''
    df = pd.read_csv('datasets/'+symbol+'.csv', parse_dates=['Date'])
    df = df.set_index('Date') #sets index of the dataframe as the Date column instead of ordinal numbering.
    return df

def splitDataset(X, y):
    '''Splits the data into Training and Test/Dev set. 80% - Train.
    Though the function train_test_split() could have been called as and when required in any script,
    It has been included in this file only to maintain the format and utility of scripts.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def addFeatures(df):
    '''Following features have been considered:
    1. High-Low: It is the difference between High and Low prices of a stock for a particular day.
    2. PCT_change: It calculates the percent change shift on 5 days.
    3. MDAV5: It is the Rolling Mean Window calculation for 5 days.
    4. Return Out: Shifts the Adj. Close for stock prices by 1 day.

    Change: It is the difference between the ReturnOut and Adj. Close for a day.
            Indicates the rise/fall of the stock price for a day wrt the previous day.
    '''
    df['High-Low'] = df['High']-df['Low']
    df['PCT_change'] = df['Adj. Close'].pct_change(5)
    df['MDAV5'] = (df.loc[:,'Close']).rolling(window=5).mean()

    df['ReturnOut'] = df['Adj. Close'].shift(-1)
    df = df.dropna()
    df.loc[:, 'Change'] = df.loc[:, 'ReturnOut'] - df.loc[:, 'Adj. Close'] > 0
    X = df.loc[:, 'Adj. Close':'MDAV5']
    y = df.loc[:, 'Change']
    return [X, y]

def featureScaling(train, test):
    '''Feature scaler fits on the training set and applies transformation on the train and test set, for uniformity in data.'''
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return [train, test]
