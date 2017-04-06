"""
@author: hugoperrier
Get a list of the NASDAQ 100 comapnies by web scrapping or read it from a file
"""

import urllib
import re 
import numpy as np
import pandas as pd


# Get list of NASDAQ companies from the web
def get_NASDAQ_fromWeb(save = None):

	# Dict with company names and tickers
	ticker_dict = {}

	# Open url and find the company names in it
	for line in re.findall('\[\"[A-Z]+\".*?\",', urllib.urlopen('http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx').read()):
		if (line.find("PRESENT") < 0):		
			ticker_dict[re.findall('\".*?\"', line)[0].replace('"','')] = re.findall('\".*?\"', line)[1].replace('"','')

	# Number of companies found
	print len(ticker_dict)

	# Create 2 lists for company symbols and company names
	symbols, names = np.array(sorted(list(ticker_dict.items()), key = lambda x: x[0])).T

	#Save to file	
	if (save != None):
		df = pd.DataFrame(names, index=symbols)
		df.to_csv('NASDAQ.csv', header=False, index=True)
	
	return symbols, names

# Get list from a file
def get_NASDAQ_fromFile():

	# Open and read file
	colnames = ["symbols", "names"]
	dataNasdaq = pd.read_csv("NASDAQ.csv", names=colnames)

	# Cleanup company names
	dataNasdaq["symbols"] = dataNasdaq["symbols"].apply(lambda d: d.replace('"', ""))
	dataNasdaq["names"] = dataNasdaq["names"].apply(lambda d: d.replace('"', ""))

	# Create 2 lists for company symbols and company names
	symbols = dataNasdaq.symbols.tolist()
	names = dataNasdaq.names.tolist()
	
	return symbols, names
