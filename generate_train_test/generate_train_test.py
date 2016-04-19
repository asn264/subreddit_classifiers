'''
Sample new dataset from sqlite database in proportion to subreddit prevalence in database. 
Make sure to drop empty comments, deleted comments, and comments with 3 words or less.
'''

from sklearn.cross_validation import train_test_split
import pandas as pd
import sqlite3
import pickle


def get_sampled_data():

	#2.5 million rows
	total_sample_size = 2500000

	subreddit_data = pd.read_csv('../exploratory_analysis/subreddit_count.tsv',delimiter='\t')
	top_250 = subreddit_data[:250][[1,3]]
	sum_percents = top_250['percents'].sum(index=1)
	top_250['percents'] = top_250['percents']/sum_percents*total_sample_size

	dfs = []
	with sqlite3.connect('../exploratory_analysis/database.sqlite') as conn:

		print 'Opened connection'

		cursor = conn.cursor()

		for i in range(len(top_250)):
			
			new_df = pd.read_sql_query("SELECT subreddit,body FROM May2015 WHERE length(body)>0 AND body <> '[deleted]' AND (LENGTH(body)-LENGTH(REPLACE(body, ' ', ''))+1)>3 AND subreddit='"+top_250.ix[i,0] +"' LIMIT "+str(int(top_250.ix[i,1])), conn)
			dfs.append(new_df)
			if i%10==0:
				print 'Done querying subreddit ', i

	tot_df = pd.DataFrame()
	print 'Appending final df...'
	tot_df = tot_df.append(dfs, ignore_index=True)
	pickle.dump(tot_df, open("sampled_df.p", "wb"))


def clean(comment):

	comment = comment.encode('ascii', 'ignore')

	#Remove newline and tab characters
	comment = comment.replace('\n', '') 
	comment = comment.replace('\t', '')

	#Remove unwanted characters
	comment = comment.translate(None, '${}()[].,:;+-*/&|<>=~"')

	return comment


def get_cleaned_train_test_split():

	data = pickle.load( open("sampled_df.p",'rb'))

	X = data.values[:,1]
	y = data.values[:,0]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


	for i in range(len(X_train)):

		#Do data cleaning
		comment = X_train[i]

		#Set the current value to the cleaned comment
		X_train[i] = clean(comment)


	for i in range(len(X_test)):

		#Do data cleaning
		comment = X_test[i]

		#Set the current value to the cleaned comment
		X_test[i] = clean(comment)

	#Save to pickles
	pickle.dump(X_train, open("X_train.p", "wb"))
	pickle.dump(X_test, open("X_test.p", "wb"))
	pickle.dump(y_train, open("y_train.p", "wb"))
	pickle.dump(y_test, open("y_test.p", "wb"))
