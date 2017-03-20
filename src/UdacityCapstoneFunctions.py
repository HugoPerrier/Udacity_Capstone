import time
import pandas as pd
import numpy as np
import quandl
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

# Quandl Key : MMRL4pygXxizDk5RozgX
quandl.ApiConfig.api_key = "MMRL4pygXxizDk5RozgX"

# Class
class dataset:
        def __init__(self, name = None, start = '0000-00-00', end = '0000_00_00'):
                self.name = name
                self.start_date = start
                self.end_date = end
                self.cpmny_tick_list = []
                self.df = []
		self.features = []
		self.trueNDX = []
		self.train_score = 0.0
		self.cv_score = 0.0

        def get_cmpny_list(self, list):
                self.cmpny_tick_list = list

        def get_Quandl_data(self):
                data_list = []
                for tick in self.cmpny_tick_list:
                        tick = "WIKI/" + tick
			print tick,
                        data_list.append(quandl.get(tick, start_date=self.start_date, end_date=self.end_date))
        		data_list[-1] = calculate_variation(data_list[-1])
 	                data_list[-1] = renameFeatures(data_list[-1])
                self.df = joinDataSets(data_list, self.cmpny_tick_list)
                self.df.fillna(method='ffill', inplace=True)

	def get_NDX_data(self):
		data = web.DataReader("^NDX", 'yahoo', self.start_date, self.end_date) #['Adj Close']
		data.drop([x for x in data.columns if x.find("Adj Close") < 0], axis=1, inplace=True)
	        dict = {"Adj Close" : "NDX_AdjClose"}
        	data.rename(columns=dict, inplace = True)
		#data.rename("NDX_AdjClose")
		if (len(self.df) == len(data)):
			self.df = pd.concat([self.df, data], axis=1)
		else:
			data.drop(data.index[[data.index.get_loc(x) for x in data.index if x not in self.df.index]], inplace=True)
			print len(data), len(self.df)
			self.df = pd.concat([self.df, data], axis=1)
			print "MEHH"
	
	def shift_data(self, df, nHist, deltaT):
	        ret_data = df
		feature_list = [x for x in df.columns]
        	for i in range (len(feature_list)):
                	for j in range (1, nHist + 1):
				shift = j + deltaT - 1
                        	histFeature = self.df[feature_list[i]].shift(shift)
                        	histFeature = histFeature.rename("%sT%d" %(feature_list[i],shift))
                        	ret_data = ret_data.join(histFeature)
			ret_data.drop([feature_list[i]], axis=1, inplace=True)
	        ret_data = ret_data[(nHist+deltaT-1):]
        	return ret_data

	def create_features(self, feat, nHist=1, deltaT = 1):
		self.trueNDX = self.df["NDX_AdjClose"][(nHist+deltaT-1):]
		drop_list = [x for x in self.df.columns if not any("_%s" %y in x for y in feat)]
		drop_list.append("NDX_AdjClose")
		self.features = self.shift_data(self.df.drop(drop_list, axis=1), nHist, deltaT)

	def train_model(self, model):
		X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(self.features, self.trueNDX, train_size = 0.8, random_state=42)

	        trainModelStart = time.time()
		model.fit(X_train, Y_train)
       		print "%-56s %fs" %("Fit model in :", time.time() - trainModelStart)

        	scoreModelTrainStart = time.time()
        	self.train_score = model.score(X_train, Y_train)
        	print "%-40s %f Time : %fs" %("Score on train set :", self.train_score, time.time() - scoreModelTrainStart)
		scoreModelCVStart = time.time()
        	self.cv_score = model.score(X_valid, Y_valid)
        	print "%-40s %f Time : %fs" %("Score on cross validation set :", self.cv_score, time.time() - scoreModelCVStart)	
		return model

	def predict_NDX(self, model):
		self.predictNDX = pd.Series(model.predict(self.features), name="NDX_pred", index=self.trueNDX.index)
        	score = model.score(self.features, self.trueNDX)
                print "%-40s %f" %("Score on %s set : " %self.name, score)
		return score

	def plot(self, true, pred, save=None):
		df = pd.DataFrame(true)
		df = df.join(pred)
		fig = plt.figure()
		ax = fig.add_subplot(111)
        	df.NDX_AdjClose.plot(label="True Value")
 		df.NDX_pred.plot(label="Prediction")
		if (save != None):
			plt.savefig("%s_%s.png" %(save, self.name))
       		plt.legend()
       		plt.show()

# Functions
def calculate_variation(df):
        df["Variation"] = df["Close"].sub(df["Open"])
        return df

def renameFeatures(df):
        dict = {"Adj. Close" : "AdjClose", "Adj. Open" : "AdjOpen", "Adj. High" : "AdjHigh", "Adj. Low" : "AdjLow", "Adj. Volume" : "AdjVolume"}
        df.rename(columns=dict, inplace = True)
        return df

def joinDataSets(dfList, symbolList):
        for i in range(len(dfList)):
                dfList[i].columns = ["%s_%s"%(symbolList[i], x) for x in dfList[i].columns]
        df = pd.concat([x for x in dfList], axis=1)
        #print df['DISCA_Open']
        return df



