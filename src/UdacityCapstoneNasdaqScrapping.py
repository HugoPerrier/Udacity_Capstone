"""
@author: hugoperrier
Get a list of the NASDAQ 100 comapnies by web scrapping or read it from a file
"""

import urllib
import re 
import numpy as np
import pandas as pd


def get_NASDAQ_fromWeb(save = None):
	ticker_dict = {}

	for line in re.findall('\[\"[A-Z]+\".*?\",', urllib.urlopen('http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx').read()):
		if (line.find("PRESENT") < 0):		
			ticker_dict[re.findall('\".*?\"', line)[0].replace('"','')] = re.findall('\".*?\"', line)[1].replace('"','')

	print len(ticker_dict)

	
	symbols, names = np.array(sorted(list(ticker_dict.items()), key = lambda x: x[0])).T

	#Save to file	
	if (save != None):
		df = pd.DataFrame(names, index=symbols)
		df.to_csv('NASDAQ.csv', header=False, index=True)
	
	return symbols, names

def get_NASDAQ_fromFile():
	colnames = ["symbols", "names"]
	dataNasdaq = pd.read_csv("NASDAQ.csv", names=colnames)
	dataNasdaq["symbols"] = dataNasdaq["symbols"].apply(lambda d: d.replace('"', ""))
	dataNasdaq["names"] = dataNasdaq["names"].apply(lambda d: d.replace('"', ""))
	symbols = dataNasdaq.symbols.tolist()
	names = dataNasdaq.names.tolist()
	
	return symbols, names
