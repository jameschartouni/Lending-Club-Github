#James Chartouni and Brian Li
'''
'Charged Off':0 
'Default' :1
'Fully Paid':2 
'In Grace Period':3 
'Late (16-30 days)':4
'Late (31-120 days)':5
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn import svm, linear_model
from sklearn.preprocessing import LabelBinarizer, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV   #Perforing grid search
#classification benchmarking
from sklearn.metrics import mean_squared_error,confusion_matrix
from sklearn.metrics import precision_score, precision_recall_curve, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
import pickle




DATA_PATH = "data/"
def load_data(path = DATA_PATH):
	csv_path = os.path.join(path,"partial_consolidated_loans_cleaner.csv")
	data = pd.read_csv(csv_path)
	with open('data.pkl', 'wb') as fid:
		pickle.dump(data, fid) 
	return data 

def LinearRegression():
	pass

def PolynomialClassifier():
	pass

def AdaBoostCLF():
	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
	clf.fit(X_train, y_train)
	scores = cross_val_score(clf, X_train, y_train, cv=5)
	print("AdaBoostCLF Cross Val Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return clf 

def GradientBoostingCLF():
	clf = GradientBoostingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, learning_rate=0.5)
	clf.fit(X_train, y_train)
	scores = cross_val_score(clf, X_train, y_train, cv=5)
	print("GradientBoostingCLF Cross Val Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return clf

def RandomForestCLF():
	clf = RandomForestClassifier(n_estimators=100, n_jobs=-1) 
	clf.fit(X_train,y_train)
	scores = cross_val_score(clf, X_train, y_train, cv=5)
	print("RF Cross Val Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return clf 

def tune_RF():
	clf = RandomForestClassifier(random_state=30)
	param_grid = { "n_estimators"      : [250, 300],
	           "criterion"         : ["gini", "entropy"],
	           "max_features"      : [3, 5],
	           "max_depth"         : [10, 50],
	           "min_samples_split" : [2, 4] ,
	           "bootstrap": [True, False]}
	grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=3, scoring=rmspe)
	grid_search.fit(X_train, y_train)
	print(grid_search.best_params_)

class Never_Default_Classifier(BaseEstimator):
	def fit(self, X, y=None):
		pass
	def predict(self, X):
		return np.zeros((len(X), 3), dtype=bool)

def never_default_clf():
	clf = Never_Default_Classifier()
	scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
	print("scores: " + str(scores))
	print("Never_Default Cross Val Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#benchmarks the classification algo
def confusion_matrix_implemented(clf):
	
	#train_predictions = cross_val_predict(clf, X_train,y_train,cv=5)
	test_predictions = cross_val_predict(clf, X_test,y_test,cv=5)

	#confusion matrix 
	conf_mx = confusion_matrix(y_test,test_predictions)
	print(conf_mx)
	plt.matshow(conf_mx, cmap=plt.cm.gray)
	plt.show()

	#confusion matrix erros graph 
	row_sums = conf_mx.sum(axis=1, keepdims=True)
	norm_conf_mx = conf_mx / row_sums
	np.fill_diagonal(norm_conf_mx, 0)
	plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
	plt.show()

	print("precision score: " + str(precision_score(y_test, test_predictions, average="weighted")))
	print("recall_score: " + str(recall_score(y_test, test_predictions, average="weighted")))
	print("f1 score: " + str(f1_score(y_test, test_predictions, average="weighted")))

#benchmarks the classification algo
def confusion_matrix_implemented_binary(clf):
	test_predictions = cross_val_predict(clf, X_test_2,y_test_2,cv=5)

	#confusion matrix 
	conf_mx = confusion_matrix(y_test_2,test_predictions)
	print(conf_mx)
	plt.matshow(conf_mx, cmap=plt.cm.gray)
	plt.show()

	#confusion matrix erros graph 
	row_sums = conf_mx.sum(axis=1, keepdims=True)
	norm_conf_mx = conf_mx / row_sums
	np.fill_diagonal(norm_conf_mx, 0)
	plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
	plt.show()

	print("precision score: " + str(precision_score(y_test_2, test_predictions, average="weighted")))
	print("recall_score: " + str(recall_score(y_test_2, test_predictions, average="weighted")))
	print("f1 score: " + str(f1_score(y_test_2, test_predictions, average="weighted")))


def feature_scaling(data):
	num_pipeline = Pipeline([
		('std_scaler', StandardScaler()),
	])
	scaled_data = num_pipeline.fit_transform(data)
	return scaled_data
	pass

def dimensionality_reduction(X):
	pca = PCA(n_components=0.95)
	X_reduced = pca.fit_transform(X)
	print(pca.explained_variance_ratio_)
	return X_reduced

def plot_roc_curve(fpr, tpr, label=None):
	plt.plot(fpr, tpr, linewidth=2, label=label)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])

#print(np.where(pd.isnull(data))[1]) # checks for empty cells 
def prepareData():
	global data 
	data.drop(["id"],1,inplace=True)
	y = np.array(data['loan_status_encoded'])
	X = np.array(data.drop(['loan_status_encoded'], 1))
	X = feature_scaling(X)
	#X = dimensionality_reduction(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
	return (X_train, X_test, y_train, y_test)

def XGB_modelfit(alg, useTrainCV=True, cv_folds=4, early_stopping_rounds=50):
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		#xgtrain = xgb.DMatrix(X_train_2, label=y_train_2)
		xgtrain = xgb.DMatrix('train.buffer')
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            feval=rmspe_xg, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
		alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
	alg.fit(X_train_2, y_train_2, feval='auc')
        
    #Predict training set:
	dtrain_predictions = alg.predict(X_train_2)
	dtrain_predprob = alg.predict_proba(X_train_2)[:,1]
        
    #Print model report:
	print("\nModel Report")
	print("Accuracy : %.4g" % accuracy_score(y_train_2, dtrain_predictions))
	print("AUC Score (Train): %f" % roc_auc_score(y_train_2, dtrain_predprob))
    

def run_XGB():
	#Choose all predictors except target & IDcols
	#predictors = [x for x in train.columns if x not in [target, IDcol]]
	xgb1 = XGBClassifier(
	 learning_rate =0.5,
	 n_estimators=10,
	 max_depth=4,
	 min_child_weight=1,
	 gamma=0,
	 subsample=0.8,
	 colsample_bytree=0.8,
	 objective= 'binary:logistic',
	 n_jobs=-1,
	 scale_pos_weight=1,
	 seed=27)
	XGB_modelfit(xgb1)
	with open('XGBoost.pkl', 'wb') as fid:
		pickle.dump(xgb1, fid) 

def run_XGB_2():
	params = {
		"learning_rate":0.5,
		"n_estimators":10,
		"max_depth":4,
		"min_child_weight":1,
		"gamma":0,
		"subsample":0.8,
		"colsample_bytree":0.8,
		"objective":'binary:logistic',
		"n_jobs":-1,
		"scale_pos_weight":1,
		"seed":27
          }

	#xgtrain = xgb.DMatrix(X_train_2, label=y_train_2)
	#xgtest = xgb.DMatrix(X_test_2, label=y_test_2)
	xgtrain = xgb.DMatrix('train.buffer')
	xgtest = xgb.DMatrix('test.buffer')
	watchlist = [(xgtrain, 'train'),(xgtest, 'eval')]
	gbm1 = xgb.train(params,
		xgtrain, 
	 	evals=watchlist, 
		early_stopping_rounds=100, 
		feval=rmspe_xg,
		verbose_eval=True)
	with open('XGBoost_2.pkl', 'wb') as fid:
		pickle.dump(gbm1, fid) 

def graph_xgb_2(gbm):
	print(gbm.get_fscore())
	'''
	plot_importance(gbm)
	pyplot.show()               
	feat_imp = pd.Series(gbm.Booster().get_fscore(fmap="xgb.fmap")).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')
	'''


def graph_xgb(alg):
	#xgb.plot_importance(alg)
	#print(alg.booster().get_fscore())                
	feat_imp = pd.Series(alg.booster().get_fscore(fmap="xgb.fmap")).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')

def run_RF():
	clf = RandomForestCLF()
	# save the classifier
	with open('RF.pkl', 'wb') as fid:
		pickle.dump(clf, fid) 

def run_AdaBoost():
	clf = AdaBoostCLF()
	# save the classifier
	with open('Adaboost.pkl', 'wb') as fid:
		pickle.dump(clf, fid) 

def run_GradientBoosting():
	clf = GradientBoostingCLF()
	# save the classifier
	with open('GradientBoosting.pkl', 'wb') as fid:
		pickle.dump(clf, fid) 

def baseline_prediction():
	current_status = []
	for row in RF_clf.predict_proba(X_test):
		current_status.append(row[2])
	ones = 0
	non_ones = 0
	for row in current_status:
		if row == 1:
			ones += 1
		else:
			non_ones += 1

	print(ones)
	print(non_ones)

	ones = 0
	non_ones = 0
	for row in y_test:
		if row == 2:
			ones += 1
		else:
			non_ones += 1

	print(ones)
	print(non_ones)


# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe



def feature_importance(clf):
	for name, score in zip(list(data.columns.values), clf.feature_importances_):
		if score > .01:
			print(name, score)


#why is the false positive rate so bad?
def RF_ROC():
	y_probas_forest = cross_val_predict(RF_clf, X_train, y_train, cv = 3, method="predict_proba")
	y_scores_forest = y_probas_forest[:,1]
	fpr, tpr, thresholds = roc_curve(y_train_2, y_scores_forest)
	print("roc score: " + str(roc_auc_score(y_train_2, y_scores_forest)))
	plot_roc_curve(fpr, tpr, "Random Forest")
	plt.show()

def RF_PRC():
	y_probas_forest = cross_val_predict(RF_clf, X_train, y_train, cv = 3, method="predict_proba")
	y_scores_forest = y_probas_forest[:,1]
	precisions, recalls, thresholds = precision_recall_curve(y_train_2, y_scores_forest)
	plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
	plt.show()
	print("precision score: " + str(precision_score(y_test, test_predictions, average="weighted")))
	print("recall_score: " + str(recall_score(y_test, test_predictions, average="weighted")))
	print("f1 score: " + str(f1_score(y_test, test_predictions, average="weighted")))

def test(clf):
	y_probas = cross_val_predict(clf, X_train, y_train, cv = 3, method="predict_proba")
	scores =  clf.score(X_test, y_test)
	print("Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#------------FUNCTIONS-------------------#

#data = load_data()


with open('RF.pkl', 'rb') as fid:
	RF_clf = pickle.load(fid)

'''
with open('Adaboost.pkl', 'rb') as fid:
	Adaboost_clf = pickle.load(fid)

'''

with open('XGBoost.pkl', 'rb') as fid:
	XGBoost_clf = pickle.load(fid)

with open('XGBoost_2.pkl', 'rb') as fid:
	XGBoost_clf_2 = pickle.load(fid)


with open('data.pkl', 'rb') as fid:
	data = pickle.load(fid)

X_train, X_test, y_train, y_test = prepareData() 
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)
X_train_2 = (X_train == 2)
X_test_2 = (X_test == 2)

#load new XGBoost data
#dtrain = xgb.DMatrix(X_train_2, label=y_train_2)
#dtrain.save_binary("train.buffer")
#xgtest = xgb.DMatrix(X_test_2, label=y_test_2)
#xgtest.save_binary("test.buffer")

#-----------------RF-----------------------#
#run_RF()
#feature_importance(RF_clf)
#test(RF_clf)
#confusion_matrix_implemented(RF_clf)
#RF_ROC()
#RF_PRC()
tune_RF()

#---------------AdaBoost-----------------------#
#run_AdaBoost()
#confusion_matrix_implemented(Adaboost_clf)

#--------------GradientBoosting-----------------#
#run_GradientBoosting()

#--------------------XGBoost---------------------#
#run_XGB()
#graph_xgb(XGBoost_clf)
#confusion_matrix_implemented(XGBoost_clf)
#test(XG)

#run_XGB_2()
#graph_xgb_2(XGBoost_clf_2)

















