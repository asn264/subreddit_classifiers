'''
Sample new dataset from sqlite database in proportion to subreddit prevalence in database. 
Make sure to drop empty comments, deleted comments, and comments with 3 words or less.
'''

from sklearn.cross_validation import train_test_split
import pandas as pd
import sqlite3
import pickle
import re


def get_sampled_data():

	total_sample_size = 500000
	num_classes = 100

	subreddit_data = pd.read_csv('../exploratory_analysis/subreddit_count.tsv',delimiter='\t')
	top_classes = subreddit_data[:num_classes][[1,3]]
	sum_percents = top_classes['percents'].sum(index=1)

	#sample half of total sample size proportionally
	top_classes['percents'] = top_classes['percents']/sum_percents*total_sample_size/2

	dfs = []
	with sqlite3.connect('../exploratory_analysis/database.sqlite') as conn:

		print 'Opened connection'

		cursor = conn.cursor()

		for i in range(len(top_classes)):
			
			#sample non-null comments with more than 3 words
			new_df = pd.read_sql_query("SELECT subreddit,body FROM May2015 WHERE length(body)>0 AND body <> '[deleted]' AND (LENGTH(body)-LENGTH(REPLACE(body, ' ', ''))+1)>3 AND subreddit='"+top_classes.ix[i,0] +"' LIMIT "+str(int(top_classes.ix[i,1])+total_sample_size/2/num_classes), conn)
			dfs.append(new_df)
			if i%10==0:
				print 'Done querying subreddit ', i

	tot_df = pd.DataFrame()
	print 'Appending final df...'
	tot_df = tot_df.append(dfs, ignore_index=True)
	pickle.dump(tot_df, open("sampled_df.p", "wb"))


def clean(comment):

	comment = comment.encode('ascii', 'ignore')

	return re.sub('[^0-9a-zA-Z]+', ' ', comment)


def get_cleaned_train_test_split():

	data = pickle.load( open("sampled_df.p",'rb'))

	X = data.values[:,1]
	y = data.values[:,0]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


	#split into train and validation
	X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=0)


	for i in range(len(X_train)):

		#Do data cleaning
		comment = X_train[i]

		#Set the current value to the cleaned comment
		X_train[i] = clean(comment)


	for i in range(len(X_validate)):

		#Do data cleaning
		comment = X_validate[i]

		#Set the current value to the cleaned comment
		X_validate[i] = clean(comment)


	for i in range(len(X_test)):

		#Do data cleaning
		comment = X_test[i]

		#Set the current value to the cleaned comment
		X_test[i] = clean(comment)

	#Save to pickles
	pickle.dump(X_train, open("X_train.p", "wb"))
	pickle.dump(X_validate, open("X_validate.p", "wb"))
	pickle.dump(X_test, open("X_test.p", "wb"))
	pickle.dump(y_train, open("y_train.p", "wb"))
	pickle.dump(y_validate, open("y_validate.p", "wb"))
	pickle.dump(y_test, open("y_test.p", "wb"))

def main():
    get_sampled_data()
    get_cleaned_train_test_split()

if __name__ == "__main__":
    main()
