'''
Sample new dataset from sqlite database in proportion to subreddit prevalence in database. 
Make sure to drop empty comments, deleted comments, and comments with 3 words or less.
'''

import pandas as pd
import sqlite3

#2.5 million rows
total_sample_size = 2500000

subreddit_data = pd.read_csv('../exploratory_analysis/subreddit_count.tsv',delimiter='\t')
top_250 = subreddit_data[:250][[1,3]]
sum_percents = top_250['percents'].sum(index=1)
top_250['percents'] = top_250['percents']/sum_percents*total_sample_size


with sqlite3.connect('../exploratory_analysis/database.sqlite') as conn:
	with open('train_test_data.tsv','a') as out_f:

		print 'Opened connection'

		cursor = conn.cursor()

		for i in range(len(top_250)):
			if i%10==0:
				print i
			table = pd.read_sql_query("SELECT subreddit,body FROM May2015 WHERE length(body)>0 AND body <> '[deleted]' AND (LENGTH(body)-LENGTH(REPLACE(body, ' ', ''))+1)>3 AND subreddit='"+top_250.ix[i,0] +"' LIMIT "+str(int(top_250.ix[i,1])), conn)
			
			table.to_csv(out_f, sep='\t',header=False,encoding='utf-8',index=False)