import pandas as pd
import requests
import json
import datetime as dt
# TODO: добавить брокера

def save_data(name: str, data, extension: str):
    with open(f'data/{name}.{extension}', 'w') as file:
        if extension == 'json':
            file.write(json.dumps(data))
        elif extension == 'csv' and 'DataFrame' in str(type(data)):
            file.write(data.to_csv())
        else:
            file.write(data)


def get_general_info(file_type='json', encoding='utf-8'):
    url = f"https://iss.moex.com/iss/statistics/engines/stock/quotedsecurities.{file_type}"
    request = requests.get(url)
    request.encoding = encoding
    data = request.json()
    save_data('imoex_test', data)
    all_stocks_df = pd.DataFrame(data['quotedsecurities']['data'], columns=data['quotedsecurities']['columns'])
    return all_stocks_df


def set_unix_time(days=100):
    '''function finds ending date "days" days before today, NOT including weekends'''
    DAYSEC = 24 * 60 * 60
    # all dates are in unix format
    actual_time_unix = dt.datetime.now().timestamp()
    days_unix = days * DAYSEC
    weekends_in_seconds = int(days_unix * 2 / 7)
    total_time_back = days_unix + weekends_in_seconds
    # convert unix timestamp to string
    begin = dt.datetime.fromtimestamp(actual_time_unix - total_time_back).strftime('%Y-%m-%d')
    ending = dt.datetime.fromtimestamp(actual_time_unix).strftime('%Y-%m-%d')

    return begin, ending, total_time_back


def get_stock_history(begin_date: str, end_date: str, ticker: str, extension='json'):
    engines = {'stock': 'stock', }
    markets = {'shares': 'shares', 'bonds': 'bonds', 'foreignshares': 'foreignshares'}
    # url = f"https://iss.moex.com/iss/history/engines/{engines['stock']}/markets/{markets['shares']}/securities/MOEX.json?from={begin_date}&till={end_date}&marketprice_board=1"
    url = f"https://iss.moex.com/iss/history/engines/{engines['stock']}/markets/{markets['shares']}/securities/{ticker}.{extension}?from={begin_date}&till={end_date}&sort_order=desc&marketprice_board=1"
    print(url)
    request = requests.get(url)
    data = json.loads(request.text)
    columns = data['history']['columns']
    df = pd.DataFrame(data['history']['data'], columns=columns)
    return df


def get_daily_stock_history(ticker: str, date: str, encoding='utf-8'):
    engines = {'stock': 'stock', }
    url = f"https://iss.moex.com/iss/statistics/engines/{engines['stock']}/currentprices.json?date={date}"
    request = requests.get(url)
    request.encoding = encoding
    data = json.loads(request.text)
    save_data(f'daily{ticker}{date}', data, 'json')
    df = pd.DataFrame(data['currentprices']['data'], columns=data['currentprices']['columns'])
    save_data(f'daily{ticker}{date}', df, 'csv')
    return df


if __name__ == '__main__':
    # parse from "https://iss.moex.com/iss/" + сatalog, get from https://iss.moex.com/iss/reference
    TICKER = 'MOEX'
    columns = ['BOARDID', 'TRADEDATE', 'SHORTNAME', 'SECID', 'NUMTRADES', 'VALUE', 'OPEN', 'LOW', 'HIGH',
               'LEGALCLOSEPRICE', 'WAPRICE', 'CLOSE', 'VOLUME', 'MARKETPRICE2', 'MARKETPRICE3', 'ADMITTEDQUOTE',
               'MP2VALTRD', 'MARKETPRICE3TRADESVALUE', 'ADMITTEDVALUE', 'WAVAL', 'TRADINGSESSION', 'CURRENCYID',
               'TRENDCLSPR']
    basic_df = pd.DataFrame([], columns=columns)

    get_daily_stock_history('IMOEX', '2024-05-03')
    begin, ending, total_time_back = set_unix_time(100)
    for i in range(20): # 1 iteration +- 5 month
        df = get_stock_history(begin_date=begin, end_date=ending, ticker=TICKER)
        basic_df = pd.concat([basic_df, df])

        ending_unix = dt.datetime.strptime(begin, '%Y-%m-%d').timestamp()
        begin_unix = (ending_unix - total_time_back)

        begin = dt.datetime.fromtimestamp(begin_unix).strftime('%Y-%m-%d')
        ending = dt.datetime.fromtimestamp(ending_unix).strftime('%Y-%m-%d')

    start = dt.datetime.now().strftime('%Y-%m-%d')
    save_data(f'{TICKER}_{start}', data=basic_df, extension='csv')
