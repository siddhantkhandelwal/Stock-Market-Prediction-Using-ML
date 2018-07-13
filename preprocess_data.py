import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split

def loadDataset(symbol):
    df = pd.read_csv('datasets/'+symbol+'.csv', parse_dates=['Date'])
    df = df.dropna()
    df.set_index('Date', inplace=True)
    return df

def splitDataset(df):
    train, test = train_test_split(df, test_size=0.2)
    train = train.dropna()
    test = test.dropna()
    return [train, test]

def addFeatures(df):
    df['High-Low'] = df['High']-df['Low']
    df['PCT_change'] = (df['Close'] - df['Open'])/df['Open'] * 100
    df['WILR'] = (df['High']- df['Close'])/(df['High']- df['Low'])*100
     
    df['MAV5'] = (df.loc[:,'Close']).rolling(window =5).mean()
    df['MAV3'] = (df.loc[:,'Close']).rolling(window =3).mean()
    
    df['ReturnOut'] = df['Adj. Close'].shift(-1)
    df = df.dropna()
    X = df.loc[:, 'Adj. Close':'MAV3']
    y = df.loc[:, 'ReturnOut']
    return [X, y]

def featureScaling(df):
    [train, test] = splitDataset(df)
    scaler = MinMaxScaler()
    train.to_csv('train.csv')
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return [train, test]
