#James Chartouni & Brian Li
#merges loans into one .csv file
import pandas as pd 
import random 


def consilidate_all_data():
	LoanStats_2016Q1 = pd.read_csv("data/LoanStats_2016Q1.csv")
	LoanStats_2016Q2 = pd.read_csv("data/LoanStats_2016Q2.csv")
	LoanStats_2016Q3 = pd.read_csv("data/LoanStats_2016Q3.csv")

	LoanStats3a = pd.read_csv("data/LoanStats3a.csv")
	LoanStats3b = pd.read_csv("data/LoanStats3b.csv")
	LoanStats3c = pd.read_csv("data/LoanStats3c.csv")
	LoanStats3d = pd.read_csv("data/LoanStats3d.csv")

	frames = [LoanStats_2016Q1, LoanStats_2016Q2, LoanStats_2016Q3, LoanStats3a, LoanStats3b, LoanStats3c, LoanStats3d]
	#frames = [LoanStats_2016Q1]
	df_consolidated = pd.concat(frames)
	#df_consolidated = df_consolidated.head(n=1000)
	#df_consolidated.drop(df_consolidated.columns[1], axis=1, inplace=True)
	#df_consolidated.to_csv("data/consolidated_loans.csv", encoding='utf-8')
	df_consolidated.to_csv("data/consolidated_loans.csv", sep='\t', encoding='utf-8')
	return df_consolidated

def partial_consilidate_all_data():
	
	LoanStats3c = pd.read_csv("data/LoanStats3b.csv")
	LoanStats3d = pd.read_csv("data/LoanStats3c.csv")

	frames = [LoanStats3c, LoanStats3d]
	#frames = [LoanStats_2016Q1]
	df_consolidated = pd.concat(frames)
	#df_consolidated = df_consolidated.head(n=1000)
	#df_consolidated.drop(df_consolidated.columns[1], axis=1, inplace=True)
	df_consolidated.to_csv("data/partial_consolidated_loans.csv", encoding='utf-8')
	return df_consolidated


def practice_consilidated_data():
	df = consilidate_all_data()
	df = df.ix[random.sample(df.index, 1000)]
	df.to_csv("data/practice_consolidated_loans.csv", sep='\t', encoding='utf-8')
	return df 

def practice_data():
	#The data to load
	f = "data/consolidated_loans.csv"

	# Count the lines
	num_lines = sum(1 for l in open(f))

	# Sample size - in this case ~1%
	size = int(num_lines / 100)

	# The row indices to skip - make sure 0 is not included to keep the header!
	skip_idx = random.sample(range(1, num_lines), num_lines - size)
	print(len(skip_idx))
	# Read the data
	data = pd.read_csv(f, skiprows=skip_idx)
	data.to_csv("data/practice_consolidated_loans.csv", sep='\t', encoding='utf-8')
	return data


#function calls 
consilidate_all_data()
#partial_consilidate_all_data()
#practice_consilidated_data()
#practice_data()

