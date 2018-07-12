import quandl
import datetime
import pandas as pd
import sys

#My API Key. Replace with yours!
quandl.ApiConfig.api_key = 'SLH4M7i2DCfVc7Npr_zV'

def get_data_from_quandl(symbol, start_date, end_date):
      file_name = symbol + '.csv'
      data = quandl.get("WIKI/"+symbol,
            returns='pandas',
            start_date=start_date,
            end_date=end_date,
            collapse='daily',
            order='asc',
            )
      data.describe()
      data.to_csv('datasets/' + file_name)

start_date = '2010-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
symbol = sys.argv[1]
print('Getting Data from quandl between ' + start_date + ' and ' + end_date + ' for the stock ' + symbol)
get_data_from_quandl(symbol, start_date, end_date)
