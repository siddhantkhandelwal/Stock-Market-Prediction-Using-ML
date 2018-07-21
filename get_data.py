import quandl
import datetime
import pandas as pd
import sys


#My API Key. Replace with yours!
quandl.ApiConfig.api_key = 'SLH4M7i2DCfVc7Npr_zV'

def get_data_from_quandl(symbol, start_date, end_date):
    '''Gets daily stock data from Quandl in ascending order.
    Takes the symbols of the required stock as command line arguments.
    The start date is set to 2002-01-01
    The end date is taken from the current date and time of the system.
    The data is returned in pandas dataframe, stored in csv format in a folder 'datasets' in the cwd.
    '''
    file_name = symbol + '.csv'
    data = quandl.get("WIKI/"+symbol,
    returns='pandas',
    start_date=start_date,
    end_date=end_date,
    collapse='daily',
    order='asc',
    )
    data.to_csv('datasets/' + file_name)

start_date = '2002-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

symbols = sys.argv[1:]

for symbol in symbols:
    print('Getting Data from quandl between ' + start_date + ' and ' + end_date + ' for the stock ' + symbol)
    get_data_from_quandl(symbol, start_date, end_date)
