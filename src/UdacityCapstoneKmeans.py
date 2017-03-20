"""
@author: hugoperrier
Apply Kmeans clustering to group companies
Select a company from each cluster a create a file containing the list of selected companies
"""

# Modules and libraries
import numpy as np
import pandas as pd
import quandl
import time
import matplotlib.pyplot as plt
import urllib
import re 
from sklearn import cluster, covariance
from sklearn.cluster import KMeans

# Other files
from UdacityCapstoneFunctions import *
from UdacityCapstoneNasdaqScrapping import *

print "\n=========== Start Program ===========\n"
# Quandl Key : MMRL4pygXxizDk5RozgX
quandl.ApiConfig.api_key = "MMRL4pygXxizDk5RozgX"

# Select the start and end date of the training period
train_start_date="2013-01-01"
train_end_date="2013-12-01"

print "\n=========== Get train data ===========\n"
# Get a list of NASDAQ tickers and company names from web scrapping
symbols, names = get_NASDAQ_fromWeb()
# Or from a file
#symbols, names = get_NASDAQ_fromFile()

# Create a dictionnary to link tickers to company names
dictionary = dict(zip(symbols, names))

# Initialize arrays
Quandl_missing_list = []
Quandl_empty_list = []
Quandl_list = []
Quandl_tick_list = []

# Get the data
getDataStart = time.time()
train_data = []
initial_date = pd.to_datetime("2100-05-31")

# Print the number of element in the company list
print "Number of tickers in NASDAQ: %d" %len(symbols)

# Get the data from Quandl
for tick in symbols:
	# Select the ticker of interest
	ticker = "WIKI/" + tick

	# Errors can occur like company not in the Quandl database or empty dataframes and need to be handled
	try:
		data = (quandl.get(ticker, start_date=train_start_date, end_date=train_end_date))

		# Empty datasets are removed
		if (data.empty):
			Quandl_empty_list.append(tick)
			print "Ticker %s gives empty dataframe" %ticker
			continue

		# Make sure all datasets have the same start date
		date = pd.to_datetime(data.index.values[0])
		if (date < initial_date):
			initial_date = date
		if (date > initial_date):
			Quandl_empty_list.append(tick)
			print "Ticker %s does not start at the initial date" %ticker
			continue

		# Add new dataset to list of datasets
		else:
			train_data.append(data)
			Quandl_tick_list.append(tick)
			Quandl_list.append(dictionary[tick])
	
	# Company not in database
	except quandl.errors.quandl_error.NotFoundError:
		Quandl_missing_list.append(tick)
		print "Missing ticker %s" %ticker

# Print info on data acquisition
print "Train data obtained in %.fs" %(time.time() - getDataStart)
print "Number of tickers missing in Wiki database: %d" %len(Quandl_missing_list)
print "Number of tickers with empty database in Wiki database: %d" %len(Quandl_empty_list)
print "Number of effective tickers: %d" %len(Quandl_list)
#print "list of tickers missing in Wiki database: %s" %",".join(Quandl_missing_list)
print "list of tickers with empty database in Wiki database: %s" %",".join(Quandl_empty_list)

# Join different datasets and fill missing data
df = joinDataSets(train_data, Quandl_tick_list)
df.fillna(method='ffill', inplace=True)

# Create the Variation = Close - Open feature
openX = np.array([df[col] for col in df.columns if col.find("Adj") < 0 and col.find("Open") > 0]).astype(np.float)
closeX = np.array([df[col] for col in df.columns if col.find("Adj") < 0 and col.find("Close") > 0]).astype(np.float)
variation = closeX - openX

Quandl_list = np.asarray(Quandl_list)
Quandl_tick_list = np.asarray(Quandl_tick_list)

# Normalize data before clustering
X = variation.copy().T
X /= X.std(axis=0)


# Start clustering
nclusterMax = 20
for ncluster in range(1, nclusterMax):
	# Use Kmeans
	kmeans = KMeans(n_clusters = ncluster, random_state = 42, n_init = 50)
	kmeans.fit(X.T)
	klabels = kmeans.labels_
	knlabels = klabels.max()
	
	# Select one company from each cluster
	cluster_tick_list = []
	cluster_list = []
	for i in range(knlabels + 1):
		print('Cluster %i: %s' % ((i + 1), ', '.join(Quandl_list[klabels == i])))
		cluster_tick_list.append(Quandl_tick_list[klabels == i][0])
		cluster_list.append(Quandl_list[klabels == i][0])
	
	drop_list = [col for col in df.columns if col.split('_')[0] not in cluster_tick_list]
	df.drop(drop_list, axis=1, inplace=True)	
	dictionary = dict(zip(cluster_tick_list, cluster_list))
	
	# Print a list of selected comapnies to file
	pd.DataFrame.from_dict(dictionary, orient="index").to_csv('cluster%d_header.csv' %ncluster, header=False)
