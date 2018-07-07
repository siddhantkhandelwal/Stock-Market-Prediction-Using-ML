def getStocks(symbol, start, end):
	'''
	Downloads stocks from Yahoo Finance
	Computes daily Returns based on Adj Close.
	Returns pandas dataframe.
	'''
	df = pd.io.data.get_data_yahoo(symbol, start, end)

	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + symbol
	df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()

	return df

def getStocksfromQuandl(symbol, name, start, end):
	'''
	Downloads stocks from Quandl
	Computes daily Returns based on Adj Close.
	Returns pandas dataframe.
	'''
	import Quandl
	df = Quandl.get(symbol, trim_start=start, trim_end=end, authtoken="")
	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + name
	df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()
	return df

def getStocksDataFromWeb(fout, start_string, end_string):
	start = parser.parse(start_string)
	end = paerser.parse(end_string)

	nasdaq = getStocks('^IXIC', start, end)
	frankfurt = getStocks('^GDAXI', start, end)
	london = getStocks('^FTSE', start, end)
	paris = getStocks('^FCHI', start, end)
	hkong = getStock('^HSI', start, end)
    nikkei = getStock('^N225', start, end)
    australia = getStock('^AXJO', start, end)

    djia = getStocksfromQuandl('YAHOO/INDEX_DJI', 'Djia', start_string, end_string)

    out = pd.io.data.get_data_yahoo(fout, start, end)
    out.columns.values[-1] = 'AdjClose'
    out.columns = out.columns + '_Out'
    out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
    return [out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia]