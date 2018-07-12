import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def loadDataset(symbol):
      df = pd.read_csv('datasets/'+symbol+'.csv', parse_dates=['Date'])
      df.set_index('Date', inplace=True)
      return df

def splitDataset(df):
    length = len(df.iloc[:,0])
    train_end = round(length * 0.80)
    test_end = round(length * 0.90)
    return [
            df.iloc[: train_end, : ],
            df.iloc[train_end:test_end, :],
            df.iloc[test_end: ,:]
           ]

def addFeatures(df):
      df['High-Low'] = df['High']-df['Low']
      df['ReturnOut'] = df['Adj. Close'].shift(-1)
      df = df.dropna()
      X = df.loc[:, 'Adj. Open':'High-Low']
      y = pd.DataFrame(df, columns = ['ReturnOut'])
      return X, y

def featureScaling(df):
      train, cv, test = splitDataset(df)
      scaler = MinMaxScaler()
      scaler.fit(train)
      train = scaler.transform(train)
      cv = scaler.transform(cv)
      test.dropna(inplace=True)
      test = scaler.transform(test)
      print(test)
      return train, cv, test
