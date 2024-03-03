from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re
import csv
from pyfinviz.quote import Quote
import json
import os
from yahooquery import Ticker

class web_driver:
    def __init__(self) -> None:
        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--headless')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        self.browser = webdriver.Firefox(options=options)
        self.browser.implicitly_wait(2)

    def load_url(self, url):
        """Load the given URL."""
        self.browser.get(url)

    def get_is_valid_url(self):
        """Check if the current URL is valid."""
        if self.browser.current_url == "https://roic.ai/404":
            self.close()
            return False
        return True
    
    def close(self):
        """Close the browser."""
        self.browser.close()
        self.browser.quit()
    
    def find_element_XPATH(self, xpath):
        """Find an element by its XPATH."""
        return self.browser.find_element(By.XPATH, xpath)

def get_stock_data(url, tables):
    """Get stock data from the given URL and tables."""

    wd = web_driver()
    wd.load_url(url=url)

    data = {}

    if not wd.get_is_valid_url():
        return data
    
    for table in tables:
        try:
            ele = wd.find_element_XPATH(table)
        except:
            continue
        html_str = ele.get_attribute('innerHTML')
        bs = BeautifulSoup(html_str, 'html.parser')
        element = bs.findChildren("div", recursive=False)
        if len(element) > 1:
            element = element[1] # Skip header
            nl = {}
            titles = element.findAll('div', {'class': 'has-tooltip'})
            titles = [x.get_text() for x in titles]
            title_i = 0
            num_titles = len(titles)

            all_eles = element.findAll('div', {'class': 'w-20'})
            divider = int(len(all_eles) / num_titles)
            for n_t in range(num_titles):
                title = titles[n_t]
                for i in range(divider):
                    start = n_t * divider
                    ele_group = all_eles[start:start+divider]
                    nl[title] = [float(x.get_text().replace('%', '').replace(',', '')) if (x.get_text() != '- -' and x.get_text() != 'premium') else -1 for x in ele_group]
                    nl[title] = nl[title][:-5]
                title_i += 1
            index = re.findall(r'"([^"]*)"', table)[0]
            data[index] = nl

    wd.close()
    return data

def get_stock_financials(ticker):
    """Get stock financials for the given ticker."""
    url = f"https://roic.ai/quote/{ticker}/financials?yearRange=1000"

    tables = ['//*[@data-cy="financial_table_incomeStatement"]', 
              '//*[@data-cy="financial_table_balanceSheet"]', 
              '//*[@data-cy="financial_table_cashFlowStatement"]']
    
    return get_stock_data(url, tables)

def get_stock_ratios(ticker):
    """Get stock ratios for the given ticker."""
    url = f"https://roic.ai/quote/{ticker}/ratios?yearRange=1000"

    tables = ['//*[@data-cy="financial_table_profitabilityFinancialRatios"]', 
              '//*[@data-cy="financial_table_creditFinancialRatios"]', 
              '//*[@data-cy="financial_table_liquidityFinancialRatios"]',
              '//*[@data-cy="financial_table_workingCapitalFinancialRatios"]',
              '//*[@data-cy="financial_table_enterpriseValueFinancialRatios"]',
              '//*[@data-cy="financial_table_multiplesFinancialRatios"]',
              '//*[@data-cy="financial_table_perShareDataItemsFinancialRatios"]']

    return get_stock_data(url, tables)

def get_stock_rating(ticker):
    """Get stock rating for the given ticker."""
    stock = Ticker(ticker)
    try:
        rating = stock.financial_data[ticker]['recommendationMean']
    except:
        rating = None
    esg = stock.esg_scores[ticker]

    esg_s = {}
    if type(esg) != str:
        try:
            esg_s['totalEsg'] = esg['totalEsg']
        except:
            esg_s['totalEsg'] = None
        try:
            esg_s['environmentScore'] = esg['environmentScore']
        except:
            esg_s['environmentScore'] = None
        try:
            esg_s['socialScore'] = esg['socialScore']
        except:
            esg_s['socialScore'] = None
        try:
            esg_s['governanceScore'] = esg['governanceScore']
        except:
            esg_s['governanceScore'] = None
        
    return rating, esg_s

def get_finviz_data(ticker):

    quote = Quote(ticker=ticker)

    if not quote or not quote.exists:
        return None

    fundamental_df = quote.fundamental_df
    return_Data = {}
    try:
        return_Data['RSI'] = float(fundamental_df['RSI (14)'].values[0])
    except:
        return_Data['RSI'] = -1
    try:
        return_Data['P/E'] = float(fundamental_df['P/E'].values[0][0])
    except:
        return_Data['P/E'] = -1
    try:
        return_Data['Beta'] = float(fundamental_df['Beta'].values[0])
    except:
        return_Data['Beta'] = -1
    try:
        return_Data['PEG'] = float(fundamental_df['PEG'].values[0])
    except:
        return_Data['PEG'] = -1
    try:
        return_Data['SMA20'] = float(fundamental_df['SMA20'].values[0].replace('%',''))
    except:
        return_Data['SMA20'] = -1
    try:
        return_Data['SMA50'] = float(fundamental_df['SMA50'].values[0].replace('%',''))
    except:
        return_Data['SMA50'] = -1
    try:
        return_Data['SMA200'] = float(fundamental_df['SMA200'].values[0].replace('%',''))
    except:
        return_Data['SMA200'] = -1
    try:
        return_Data['Recom'] = float(fundamental_df['Recom'].values[0])
    except:
        return_Data['Recom'] = -1
    return return_Data
    
def save_stock_Data(path, data):
    front_str = ',{}]'
    if not os.path.isfile(path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        front_str = '{}]'
    
    with open(path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()
    
    with open (path, mode="a") as file:
        file.seek(0,2)
        position = file.tell() - 1
        file.seek(position)
        file.write(front_str.format(json.dumps(data)))

def load_tickers_done(path):
    if not os.path.isfile(path):
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        try:
            tickers_list = json.load(f)
            return tickers_list
        except ValueError:
            raise IOError("Failed to load the JSON file.")

def save_ticker(path, ticker):
    front_str = ',{}]'
    
    if not os.path.isfile(path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f)
            front_str = '{}]'

    with open(path, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        f.truncate()
    
    with open (path, mode="a") as file:
        file.seek(0,2)
        position = file.tell() - 1
        file.seek(position)
        file.write(front_str.format(json.dumps(ticker)))

CSV_PATH = '/home/me/Nextcloud/Dev/llm/nasdaq.csv' # Download from nasdaq
TICKER_PATH = '/home/me/Nextcloud/Dev/llm/tickers.json'

def load_tickers(path):
    file = open(path) 
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row[0])
    del(rows[0]) # remove header
    return rows

def main():
    tickers = load_tickers(CSV_PATH)
    stock_data = {}
    N = 5  # print progress every Nth time
    done_tickers = load_tickers_done(TICKER_PATH)

    for i, ticker in enumerate(tickers):
        if (i + 1) % N == 0:
            print(f"Processed {i + 1} out of {len(tickers)} tickers. Ticker: {ticker}")
        if ticker in done_tickers:
            continue

        ticker_Data = {}

        finviz = get_finviz_data(ticker)
        if not finviz:
            continue

        rating_out, esg_s = get_stock_rating(ticker)

        finviz_rating = finviz['Recom']
        rating = rating_out

        if not rating_out:
            rating = 0
        if not finviz['Recom']:
            finviz_rating = 0

        financials = get_stock_financials(ticker)
        ratios = get_stock_ratios(ticker)

        ticker_Data['ratings'] = (rating + finviz_rating) / 2
        ticker_Data['input'] = [finviz, esg_s, financials, ratios]
        if rating_out or finviz['Recom']:
            stock_data[ticker] = ticker_Data
            save_stock_Data('/home/me/Nextcloud/Dev/llm/stocks_data.json', stock_data)
            save_ticker(TICKER_PATH, ticker)
        stock_data = {}

if __name__ == '__main__':
    main()

