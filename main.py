import pickle
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import datetime
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import operator
from pandas_datareader import data
import re
from dateutil import parser
import sys
import matplotlib.pyplot as plt

fout = 'sp500'
method = 'RF'
best_model = 'sp500_57.pickle'

path_datasets = 'datasets'
cut = datetime.datetime(1993,1,1)
start_test = datetime.datetime(2014,4,1)
parameters = []

maxlags = 10
maxdeltas = 10
folds = 10

bestlags = 9
bestdelta = 9
savemodel = False

    

def loadDatasets(path_directory, fout): 
    """
    import into dataframe all datasets saved in path_directory
    """ 
    name = path_directory + '/' + fout + '.csv'
    out = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/nasdaq.csv'
    nasdaq = pd.read_csv(name, index_col=0, parse_dates=True)
    
    return [out, nasdaq]
    

def getStocks(symbol, start, end):
    '''
    Downloads stocks from Yahoo Finance
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    '''
    df = data.get_data_yahoo(symbol, start, end)
    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()

    return df


def getStocksDataFromWeb(fout, start_string, end_string):
    start = parser.parse(start_string)
    end = parser.parse(end_string)

    nasdaq = getStocks('^IXIC', start, end)
    
    out = pd.io.data.get_data_yahoo(fout, start, end)
    out.columns.values[-1] = 'AdjClose'
    out.columns = out.columns + '_Out'
    out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
    return [out, nasdaq]

def count_missing(dataframe):
    """
    count number of NaN in dataframe
    """
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()


def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days
 
    """

    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)

    roll_n = returns[7:] + 'RolMean' + str(n)
    #dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)
    dataframe[roll_n] = dataframe.rolling(window=n).mean()[returns]


def applyRollMeanDelayedReturns(datasets, delta):
    '''
    applies rolling mean and delayed returns to each dataframe in the list
    '''

    for dataset in datasets:
        columns = dataset.columns
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            addFeatures(dataset, adjclose, returns, n)
    return datasets

def mergeDataframes(datasets, index, cut):
    '''
    merges datasets in the list
    '''

    subset = []
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
    first = subset[0].join(subset[1:], how = 'outer')
    finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
    finance = finance[finance.index > cut]
    return finance

def checkModel(filename):
    """
    checks max accuracy after CV  with specific algorithm
    """
    txt = open(filename, "r")
    lines = txt.readlines()
    accuracies = [line[:-1] for line in lines if line.startswith('0.')]    
    txt.close()
    return  max(accuracies)


def applyTimeLag(dataset, lags, delta):
    """
    apply time lag to return columns selected according  to delta.
    Days to lag are contained in the lads list passed as argument.
    Returns a NaN free dataset obtained cutting the lagged dataset
    at head and tail
    """
    
    dataset.Return_Out = dataset.Return_Out.shift(-1)
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for lag in lags:
            newcolumn = column + str(lag)
            dataset[newcolumn] = dataset[column].shift(lag)

    return dataset.iloc[maxLag:-1,:]

def performCV(X_train, y_train, folds, method, parameters, fout, savemodel):
    """
    given complete dataframe, number of folds, the % split to generate 
    train and test set and features to perform prediction --> splits
    dataframein test and train set. Takes train set and splits in k folds.
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    returns mean of test accuracies
    """
    print ('')
    print ('Parameters --------------------------------> ', parameters)
    print ('Size train set: ', X_train.shape)
    
    k = int(np.floor(float(X_train.shape[0])/folds))
    
    print ('Size of each fold: ', k)
    
    acc = np.zeros(folds-1)
    for i in range(2, folds+1):
        print ('')
        split = float(i-1)/i
        print ('Splitting the first ' + str(i) + ' chuncks at ' + str(i-1) + '/' + str(i))
        data = X_train[:(k*i)]
        output = y_train[:(k*i)]
        print ('Size of train+test: ', data.shape)
        index = int(np.floor(data.shape[0]*split))
        X_tr = data[:index]        
        y_tr = output[:index]
        
        X_te = data[(index+1):]
        y_te = output[(index+1):]        
        
        acc[i-2] = performClassification(X_tr, y_tr, X_te, y_te, method, parameters, fout, savemodel)
        print ('Accuracy on fold ' + str(i) + ': ', acc[i-2])
    
    return acc.mean()     

def performTimeSeriesSearchGrid(X_train, y_train, folds, method, grid, fout, savemodel):
    """
    parameters is a dictionary with: keys --> parameter , values --> list of values of parameter
    """
    print ('')
    print ('Performing Search Grid CV...')
    print ('Algorithm: ', method)
    param = grid.keys()
    finalGrid = {}
    if len(param) == 1:
        for value_0 in grid[param[0]]:
            parameters = [value_0]
            accuracy = performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
            finalGrid[accuracy] = parameters
        final = sorted(finalGrid.iteritems(), key=operator.itemgetter(0), reverse=True)  
        print ('')
        print (finalGrid)
        print ('')
        print ('Final CV Results: ', final)
        return final[0]
        
    elif len(param) == 2:
        for value_0 in grid[param[0]]:
            for value_1 in grid[param[1]]:
                parameters = [value_0, value_1]
                accuracy = performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
                finalGrid[accuracy] = parameters
        final = sorted(finalGrid.iteritems(), key=operator.itemgetter(0), reverse=True)
        print ('')
        print (finalGrid)
        print ('')
        print ('Final CV Results: ', final)
        return final[0]   


def prepareDataForClassification(dataset, start_test):
    """
    generates categorical output column, attach to dataframe 
    label the categories and split into train and test
    """
    le = preprocessing.LabelEncoder

    dataset['Updown'] = dataset['Return_Out']
    dataset.Updown[dataset.Updown >=0] = 'Up'
    dataset.Updown[dataset.Updown <0] = 'Down'
    dataset.Updown = le.fit(dataset.Updown).transform(dataset.Updown)

    features = dataset.columns[1:-1]
    X = dataset[features]
    y = dataset.Updown

    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]
    
    X_test = X[X.index >= start_test]
    y_test = y[y.index >= start_test]

    return X_train, y_train, X_test, y_test

def performFeatureSelection(maxdeltas, maxlags, fout, cut, start_test, path_datasets, savemodel, method, folds, parameters):
    '''
    Performs feature selection for specific algrithm
    '''

    for maxlag in range(3, maxlags+2):
        lags = range(2, maxlag)
        print ('')
        print ('=============================================================')
        print ('Maximum time lag applied', max(lags))
        print ('')

        for maxdelta in range(3, maxdeltas+2):
            datasets = loadDatasets(path_datasets, fout)
            delta = range(2, maxdelta)
            print ('Delta days accounted: ', max(delta))
            datasets = applyRollMeanDelayedReturns(datasets, delta)
            finance = mergeDataframes(datasets, 6, cut)
            print ('Size of data frame: ', finance.shape)
            print ('Number of NaN after merging: ', count_missing(finance))
            finance = finance.interpolate(method='linear')
            print ('Number of NaN after time interpolation: ', count_missing(finance))
            finance = finance.fillna(finance.mean())
            print ('Number of NaN after mean interpolation: ', count_missing(finance))    
            finance = applyTimeLag(finance, lags, delta)
            print ('Number of NaN after temporal shifting: ', count_missing(finance))
            print ('Size of data frame after feature creation: ', dfinance.shape)
            X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test)
            print (performCV(X_train, y_train, folds, method, parameters, fout, savemodel))
            print ('')

def performParameterSelection(bestdelta, bestlags, fout, cut, start_test, path_datasets, savemodel, method, folds, parameters, grid):
    
    lags = range(2, bestlags + 1) 
    print ('Maximum time lag applied', max(lags))
    datasets = loadDatasets(path_datasets, fout)
    delta = range(2, bestdelta + 1) 
    print ('Delta days accounted: ', max(delta))
    datasets = applyRollMeanDelayedReturns(datasets, delta)
    finance = mergeDataframes(datasets, 6, cut)
    print ('Size of data frame: ', finance.shape)
    print ('Number of NaN after merging: ', count_missing(finance))
    finance = finance.interpolate(method='linear')
    print ('Number of NaN after time interpolation: ', count_missing(finance))
    finance = finance.fillna(finance.mean())
    print ('Number of NaN after mean interpolation: ', count_missing(finance))
    finance = applyTimeLag(finance, lags, delta)
    print ('Number of NaN after temporal shifting: ', count_missing(finance))
    print ('Size of data frame after feature creation: ', finance.shape)
    X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test)

    return performTimeSeriesSearchGrid(X_train, y_train, folds, method, grid, fout, savemodel)

def getPredictionFromBestModel(bestdelta, bestlags, fout, cut, start_test, path_datasets, best_model):
    """
    returns array of prediction and score from best model.
    """
    lags = range(2, bestlags + 1) 
    start_string = '1990-1-1'
    end_string = '2018-7-06'
    datasets = loadDatasets(path_datasets, fout)
    delta = range(2, bestdelta + 1) 
    datasets = applyRollMeanDelayedReturns(datasets, delta)
    finance = mergeDataframes(datasets, 6, cut)
    finance = finance.interpolate(method='linear')
    finance = finance.fillna(finance.mean())    
    finance = applyTimeLag(finance, lags, delta)
    X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test)    
    with open(best_model, 'rb') as fin:
        model = pickle.load(fin)        
        
    return model.predict(X_test), model.score(X_test, y_test)


def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
    """
    performs classification on daily returns using several algorithms (method).
    method --> string algorithm
    parameters --> list of parameters passed to the classifier (if any)
    fout --> string with name of stock to be predicted
    savemodel --> boolean. If TRUE saves the model to pickle file
    """
   
    if method == 'RF':   
        return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
    elif method == 'SVM':   
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)


def performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Random Forest Binary Classification
    """
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            pickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy


def performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    SVM binary Classification
    """
    c = parameters[0]
    g =  parameters[1]
    clf = SVC()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            pickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy

symbol = 'S&P-500'
name = path_datasets + '/sp500.csv'

prediction = getPredictionFromBestModel(9, 9, 'sp500', cut, start_test, path_datasets, 'sp500_57.pickle')[0]
print(prediction)
sys.stdout = open('path to log txt file', 'w')  
print (checkModel('path to log txt file'))    
