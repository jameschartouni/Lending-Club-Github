import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


'''
READ-ME
Authors: Brian Li, James Chartouni

This file is primarily used to clean a LendingClub dataset. The eventual purpose of this dataset is to use ML to predict default rate for LendingClub bonds.
The actual df that we cleaned is too big to send, so to demonstrate this, we have provided a much smaller dataset to help someone visualize the raw data we have, entitled 'raw_partial_consolidated_loans_demo.csv'.
The raw demo dataset has been reduced from 420k observations to 300 in the demo.

Note that this dataset is manually modified, so needs a special encoding to open as a result
'''


#Import the dataset we want.
#a=2 returns the cleaned dataset after running clean_dataset()
#a=1 returns the raw dataset before running clean_dataset()
def import_dataset(a):
	if a == 1:
		df = pd.read_csv("data/partial_consolidated_loans.csv",index_col=0)
	if a == 2:
		df = pd.read_csv("data/partial_consolidated_loans_cleaner.csv",index_col=0)
	else:
		df = pd.read_csv(a,index_col=0)
	return df


#Used to visualize the dataset
def visualize_dataset(df):
	#We're trying to get a sense of the dataset here
	#We see we have a few null values
	print(df.info(max_cols=120))
	
	#print(df["loan_status"].value_counts())
	#print(df["application_type"].value_counts())
	#print(df.describe())
	'''
	df.hist(bins = 50,figsize=(20,15))
	plt.show()
	'''


#return the cleaned dataset. Goal: remove unnecessary features and fill null values
def clean_dataset(df):
	df = df[df.member_id.notnull()]

	#We are only worried here about fully paid, charged off, late and default loans
	#current loans are still being paid back, so we're going to get rid of those because we don't know if they will default or not
	df = df[df["loan_status"] != "Current"]	

	#extracting number from 'emp_length'
	#Ex: '10+ years' -> 10
	#Ex: '<1 year' -> 1
	df['emp_length'] = df['emp_length'].str.extract('(\d+)',expand=False).astype(float)

	#Stripping characters from num values. More explanation provided at the to_strip() method
	#Ex: "60 months" --> "60"
	strip_column = {"term":" months","int_rate":"%","revol_util":"%"}
	df = to_strip(df,strip_column)

	#Changing categorical variables to lables (1,2,3,...). 
	categorical_to_binarizer = ["sub_grade","home_ownership","verification_status","purpose","addr_state","initial_list_status"]
	df = label_binarizer(df,categorical_to_binarizer)

	categorical_to_encode = ["loan_status"]
	df = label_encoder(df, categorical_to_encode)


	#Filling null values with high numbers, more explanation provided at the to_fill_na() method
	nan_high = {'mths_since_last_record':350,'mths_since_last_delinq':350,'mths_since_last_major_derog':350,'mths_since_recent_bc_dlq':350,'mths_since_recent_inq':350,
		'mths_since_recent_revol_delinq':350}
	#Filling null values with median numbers
	#We could have consolidated the dictionaries, but for clarity's sake, kept them separate
	nan_median = {'mo_sin_old_il_acct':'med','mo_sin_old_rev_tl_op':'med','mo_sin_rcnt_rev_tl_op':'med','mo_sin_rcnt_tl':'med','mths_since_recent_bc':'med','num_actv_rev_tl':'med','num_actv_rev_tl':'med',
		'num_bc_sats':'med','num_bc_tl':'med','num_il_tl':'med','num_op_rev_tl':'med','num_rev_accts':'med','num_rev_tl_bal_gt_0':'med','num_sats':'med','num_tl_120dpd_2m':'med','num_tl_30dpd':'med',
		'num_tl_90g_dpd_24m':'med','num_tl_op_past_12m':'med','pct_tl_nvr_dlq':'med','percent_bc_gt_75':'med','tot_hi_cred_lim':'med','total_bal_ex_mort':'med','total_bc_limit':'med','total_il_high_credit_limit':'med',
		'mort_acc':'med','num_accts_ever_120_pd':'med','num_actv_bc_tl':'med','bc_open_to_buy':'med','bc_util':'med','acc_open_past_24mths':'med','revol_util':'med','tot_cur_bal':'med','avg_cur_bal':'med',
		'tot_coll_amt':'med','tot_cur_bal':'med','total_rev_hi_lim':'med'}
	nan_zero = {'emp_length':0}
	df = to_fill_na(df,nan_high)
	df = to_fill_na(df,nan_median)
	df = to_fill_na(df,nan_zero)

	#not helpful, or were empty for every observation we had
	#a few features were dropped because we already added in the encoded categorical labels in line 72
	df.drop(['funded_amnt_inv','id','member_id','emp_title','grade','sub_grade','loan_status','home_ownership','verification_status',
		'issue_d','pymnt_plan','url','desc','purpose','title','zip_code','addr_state','earliest_cr_line','next_pymnt_d',
		'last_credit_pull_d','application_type','annual_inc_joint','dti_joint','verification_status_joint','open_acc_6m',
		'open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
		'max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m','last_pymnt_d','last_pymnt_amnt','next_pymnt_d',
		'last_credit_pull_d','collections_12_mths_ex_med','policy_code','initial_list_status', 'out_prncp', 'out_prncp_inv'
		'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'],1,inplace=True)
	
	df.to_csv("data/partial_consolidated_loans_cleaner.csv")
	
#return the cleaned dataset. Goal: remove unnecessary features and fill null values
def clean_dataset_investable_loans(df):
	df = df[df.member_id.notnull()]

	#extracting number from 'emp_length'
	#Ex: '10+ years' -> 10
	#Ex: '<1 year' -> 1
	df['emp_length'] = df['emp_length'].str.extract('(\d+)',expand=False).astype(float)


	#Changing categorical variables to lables 
	categorical_to_binarizer = ["sub_grade","home_ownership","is_inc_v","purpose","addr_state","initial_list_status"]
	df = label_binarizer(df,categorical_to_binarizer)

	#Filling null values with high numbers, more explanation provided at the to_fill_na() method
	nan_high = {'mths_since_last_record':350,'mths_since_last_delinq':350,'mths_since_last_major_derog':350,'mths_since_recent_bc_dlq':350,'mths_since_recent_inq':350,
		'mths_since_recent_revol_delinq':350}
	#Filling null values with median numbers
	#We could have consolidated the dictionaries, but for clarity's sake, kept them separate
	nan_median = {'mo_sin_old_il_acct':'med','mo_sin_old_rev_tl_op':'med','mo_sin_rcnt_rev_tl_op':'med','mo_sin_rcnt_tl':'med','mths_since_recent_bc':'med','num_actv_rev_tl':'med','num_actv_rev_tl':'med',
		'num_bc_sats':'med','num_bc_tl':'med','num_il_tl':'med','num_op_rev_tl':'med','num_rev_accts':'med','num_rev_tl_bal_gt_0':'med','num_sats':'med','num_tl_120dpd_2m':'med','num_tl_30dpd':'med',
		'num_tl_90g_dpd_24m':'med','num_tl_op_past_12m':'med','pct_tl_nvr_dlq':'med','percent_bc_gt_75':'med','tot_hi_cred_lim':'med','total_bal_ex_mort':'med','total_bc_limit':'med','total_il_high_credit_limit':'med',
		'mort_acc':'med','num_accts_ever_120_pd':'med','num_actv_bc_tl':'med','bc_open_to_buy':'med','bc_util':'med','acc_open_past_24mths':'med','revol_util':'med','tot_cur_bal':'med','avg_cur_bal':'med',
		'tot_coll_amt':'med','tot_cur_bal':'med','total_rev_hi_lim':'med'}
	nan_zero = {'emp_length':0}
	df = to_fill_na(df,nan_high)
	df = to_fill_na(df,nan_median)
	df = to_fill_na(df,nan_zero)

	loan_ids = df['id', 'exp_default']
	loan_ids.to_csv("data/lc_investable_securities_id.csv")

	#check what you want to do with service_fee_rate

	#not helpful, or were empty for every observation we had
	#a few features were dropped because we already added in the encoded categorical labels in line 72
	df.drop(['id', 'exp_default', 'service_fee_rate', 'member_id','emp_title','grade','sub_grade','loan_status','home_ownership','is_inc_v',
		'accept_d','exp_d', 'list_d', 'credit_pull_d', 'review_status_d', 'review_status', 'url','desc','purpose','title','zip_code','addr_state','msa', 
		'acc_now_delinq', 'acc_open_past_24mths', 'bc_open_to_buy', 'percent_bc_gt_75', 'bc_util', 'delinq_amnt', 'earliest_cr_line',
		'fico_range_low', 'fico_range_high', 'next_pymnt_d', 'mths_since_recent_revol_delinq', 'mths_since_recent_bc', 'mort_acc', 'open_acc', 'total_bal_ex_mort',
		'total_bc_limit', 'total_il_high_credit_limit', 'num_rev_accts', 'mths_since_recent_bc_dlq',
		'last_credit_pull_d','application_type','annual_inc_joint','dti_joint','verification_status_joint','open_acc_6m',
		'open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
		'max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m','last_pymnt_d','last_pymnt_amnt','next_pymnt_d',
		'last_credit_pull_d','collections_12_mths_ex_med','policy_code','initial_list_status'],1,inplace=True)
	
	df.to_csv("data/lc_investable_securities.csv")

#takes a list of column name strings, returns a df with changed columns
#columns are categorical and changed to categorical labels
def label_encoder(df,column_names):
	for column_name in column_names:
		encoder = LabelEncoder()
		a = df[column_name]
		a_encoded = encoder.fit_transform(a)
		df[column_name+"_encoded"] = a_encoded
		print(encoder.classes_)
	return df


#takes a list of column name strings, returns a df with changed columns
#columns are categorical and changed to categorical labels
def label_binarizer(df,column_names):
	for column_name in column_names:
		encoder = LabelBinarizer()
		a = df[column_name]
		a_encoded = encoder.fit_transform(a)
		a_encoded = a_encoded.transpose()
		i = 0
		for col in a_encoded:
			df[column_name+"_encoded" +  str(i)] = col
			i += 1
		#print(list(df))
	return df



#ex: '60 months' --> 60
#converts columns by stripping particular strings and converting resulting string number to float or int
#takes a dictionary of features to strip. Key: the feature, value: the string or character that needs to be deleted
def to_strip(df,strip_dict):
	for key,value in strip_dict.items():
		df[key] = df[key].str.strip(value)
		df[key] = pd.to_numeric(df[key], errors='coerce')
	return df

#takes a dictionary of features with NaNs. Key: feature with NaN values, value: values to replace NaNs
#Value can take three different values: 0, 'med' or 350. 'med' means we fill the NaN with median value of the feature
#350 was used because it was a high number. Ex: "mths_since_last_record" denotes the months since an arrest of some sort. NaNs denoted a clean bill
#350 just is used to show that this individual was never booked
def to_fill_na(df,nan_dict):
	for key,value in nan_dict.items():
		if value == 'med':
			df[key].fillna(value=df[key].median(),inplace=True)
		else:
			df[key].fillna(value=value,inplace=True)
	return df


#####Cleaning Dataset#####
#df = import_dataset(1)
#df = import_dataset(2)
df = import_dataset("data/primaryMarketNotes_browseNotes_1-RETAIL.csv")
#visualize_dataset(df)
#clean_dataset(df)
clean_dataset_investable_loans(df)
