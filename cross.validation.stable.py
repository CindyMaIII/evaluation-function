#!/usr/bin/python
import numpy as np
import xgboost as xgb
import sys
import json
import pandas as pd
import time

class Controller:

	reload(sys)
	sys.setdefaultencoding('utf8')

	def __init__(self, data, columnsNum, _nfold, _seed):
		self.nfold = _nfold
		self.seed = _seed

		sz = data.shape

		train = data[:int(sz[0] * 0.7), :]
		test = data[int(sz[0] * 0.7):, :]

		train_X = train[:, 0:(columnsNum-1)]
		train_Y = train[:, columnsNum]
		test_X = test[:, 0:(columnsNum-1)]
		test_Y = test[:, columnsNum]

		self.dtrain = xgb.DMatrix(train_X, label=train_Y)
		self.xg_test = xgb.DMatrix(test_X, label=test_Y)

	def find_num_round(self, param, num_round, outputFile_path):
		cvData = xgb.cv(param, self.dtrain, num_round, nfold=self.nfold, metrics={'mlogloss'}, seed=self.seed)

		test_m = [cvData.get('test-mlogloss-mean')[x] for x in range(len(cvData.get('test-mlogloss-mean')))]
		test_s = [cvData.get('test-mlogloss-std')[x] for x in range(len(cvData.get('test-mlogloss-mean')))]
		train_m = [cvData.get('train-mlogloss-mean')[x] for x in range(len(cvData.get('test-mlogloss-mean')))]
		train_s = [cvData.get('train-mlogloss-std')[x] for x in range(len(cvData.get('test-mlogloss-mean')))]

		raw_data = {'test-mlogloss-mean': test_m, 'test-mlogloss-std': test_s, 'train-mlogloss-mean': train_m, 'train-mlogloss-std': train_s}
		df = pd.DataFrame(raw_data, columns=['test-mlogloss-mean', 'test-mlogloss-std', 'train-mlogloss-mean', 'train-mlogloss-std'])
		df.to_csv(outputFile_path)
		dfsort = df.sort_values(['test-mlogloss-mean'], ascending=True)
		return list(dfsort.index)[0]+1

	def find_eta(self, param, num_round, tune_eta_parameters, outputFile_path):
		test_m = list()
		test_s = list()
		train_m = list()
		train_s = list()
		for p in tune_eta_parameters:
			param['eta']=p
			cvData = xgb.cv(param, self.dtrain, num_round, nfold=self.nfold, metrics={'mlogloss'}, seed=self.seed)
			test_m.append(cvData.get('test-mlogloss-mean')[num_round - 1])
			test_s.append(cvData.get('test-mlogloss-std')[num_round - 1])
			train_m.append(cvData.get('train-mlogloss-mean')[num_round - 1])
			train_s.append(cvData.get('train-mlogloss-std')[num_round - 1])
		raw_data = {'test-mlogloss-mean': test_m, 'test-mlogloss-std': test_s, 'train-mlogloss-mean': train_m, 'train-mlogloss-std': train_s}
		df = pd.DataFrame(raw_data, columns=['test-mlogloss-mean', 'test-mlogloss-std', 'train-mlogloss-mean', 'train-mlogloss-std'])
		df.to_csv(outputFile_path)

		dfsort = df.sort_values(['test-mlogloss-mean'], ascending=True)
		return tune_eta_parameters[list(dfsort.index)[0]]

	# param, num_round, tune_max_depth_parameters, outputFile_path
	def find_max_depth(self, param, num_round, tune_max_depth_parameters, outputFile_path):
		test_m = list()
		test_s = list()
		train_m = list()
		train_s = list()
		for p in tune_max_depth_parameters:
			param['max_depth'] = p
			cvData = xgb.cv(param, self.dtrain, num_round, nfold=self.nfold, metrics={'mlogloss'}, seed=self.seed)
			test_m.append(cvData.get('test-mlogloss-mean')[num_round - 1])
			test_s.append(cvData.get('test-mlogloss-std')[num_round - 1])
			train_m.append(cvData.get('train-mlogloss-mean')[num_round - 1])
			train_s.append(cvData.get('train-mlogloss-std')[num_round - 1])
		raw_data = {'test-mlogloss-mean': test_m, 'test-mlogloss-std': test_s, 'train-mlogloss-mean': train_m, 'train-mlogloss-std': train_s}
		df = pd.DataFrame(raw_data, columns=['test-mlogloss-mean', 'test-mlogloss-std', 'train-mlogloss-mean', 'train-mlogloss-std'])
		df.to_csv(outputFile_path)

		dfsort = df.sort_values(['test-mlogloss-mean'], ascending=True)
		return tune_max_depth_parameters[list(dfsort.index)[0]]

# >> running cross validation
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob', 'num_class':6, 'eval_metric':'mlogloss'}
# num_round = 2

# print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
# xgb.cv(param, dtrain, num_round, nfold=5, metrics={'mlogloss'}, seed = 0, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])


if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf8')

	print '------ start ------', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	columnsNum = 1804
	num_class = 9
	_nfold = 5
	_seed = 3
	_dataSrc = 'LargeTrain.csv'
	_output_folderName = 'outputdata.v3.2-1'
	# data = np.loadtxt('dermatology.data', delimiter=',', converters={(columnsNum-1): lambda x: int(x == '?'), columnsNum: lambda x: int(x) - 1})
	data = np.loadtxt(_dataSrc, delimiter=',', converters={(columnsNum-1): lambda x: int(x == '?'), columnsNum: lambda x: int(x) - 1})

	controller = Controller(data,columnsNum, _nfold, _seed)
	param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'multi:softprob', 'num_class': num_class, 'eval_metric': 'mlogloss'}
	# param = { 'silent': 1, 'objective': 'multi:softprob', 'num_class': num_class, 'eval_metric': 'mlogloss'}

	print '-------find num_round-------', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	outputFile_path = _output_folderName+'/num_round.csv'
	tune_num_round = 500
	num_round = controller.find_num_round(param, tune_num_round, outputFile_path)
	print 'num_round : ',num_round

	print '-------find eta-------', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	outputFile_path = _output_folderName+'/eta.csv'
	tune_eta_parameters = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
	eta = controller.find_eta(param, num_round, tune_eta_parameters, outputFile_path)
	print 'eta : ',eta

	print '-------find  max_depth-------', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	outputFile_path = _output_folderName+'/max_depth.csv'
	param['eta'] = eta
	tune_max_depth_parameters = [x for x in range(10)]
	max_depth = controller.find_max_depth(param, num_round, tune_max_depth_parameters, outputFile_path)
	print 'max_depth : ',max_depth

	print '______finish_________', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

