import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split

def loadDataset(symbol):
    df = pd.read_csv('datasets/'+symbol+'.csv', parse_dates=['Date'])
    df = df.set_index('Date')
    return df

def splitDataset(X, y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test

def addFeatures(df):
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
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return [train, test]
