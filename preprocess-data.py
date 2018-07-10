import quandl
import datetime
import pandas as pd

quandl.ApiConfig.api_key = 'mCDHcSdN9mQ_Hubid1Uq'

def get_data_from_quandl(symbol, start_date, end_date):
	query_list = ['WIKI' + '/' + symbol + '.' + str(k) for k in range(1, 13)]
	file_name = symbol + '.csv'
	data = quandl.get(query_list, 
            returns='pandas', 
            start_date=start_date,
            end_date=end_date,
            collapse='daily',
            order='asc'
            )
	data.to_csv(file_name, index=False)
	
start_date = '2018-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
get_data_from_quandl('AAPL', start_date, end_date)
