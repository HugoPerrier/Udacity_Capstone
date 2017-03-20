import numpy as np
import pandas as pd
import quandl
import time
import matplotlib.pyplot as plt
from UdacityCapstoneFunctions import *

# Get a list of companies and tickers
df = pd.read_csv('Companies.csv', names=['Ticker', 'Name'], comment='#')
dict=zip(list(df.Ticker), list(df.Name))
ticker_list = df.Ticker.tolist()
cmpny_list = df.Name.tolist()

# Print the list of comapnies
print ",".join(cmpny_list)

# Define train and test sets
train = dataset("Train", '2014-01-01', '2015-12-31')
test = dataset("Test", '2016-01-01', '2016-12-31')

# Set the cmpny used for feature data
train.get_cmpny_list(df.Ticker.tolist())
test.get_cmpny_list(df.Ticker.tolist())

# Get the cmpny data from Quandl
train.get_Quandl_data()
test.get_Quandl_data()

# Get the NASDAQ 100 data (NDX)
train.get_NDX_data()
test.get_NDX_data()

# Choose the features to be used for preditions
feature_choice = ["AdjOpen", "AdjClose"]

# Choose how many days are used for prediction
nHist = 3

# Divide train data into training set and cross validation set
train.create_features(feature_choice , nHist)
test.create_features(feature_choice , nHist)

# Choose a ML model
model = SVR(kernel='linear', C=1.0, epsilon=1.0)

# Get model from train data
model = train.train_model(model)

# Use model to predict test data
train.predict_NDX(model)
test.predict_NDX(model)
train.plot(train.trueNDX, train.predictNDX)
test.plot(test.trueNDX, test.predictNDX)
