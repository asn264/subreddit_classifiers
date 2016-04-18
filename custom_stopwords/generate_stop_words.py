'''
Sample new dataset from sqlite database in proportion to subreddit prevalence in database. 
Make sure to drop empty comments as well as comments that are not in English characters. 
Do word counts using Collections class.
'''

import pandas as pd
import sqlite3

total_sample_size = 10**6

subreddit_data = pd.read_csv('../exploratory_analysis/subreddit_count.tsv',delimiter='\t')
top_250 = subreddit_data[:250][[1,3]]
sum_percents = top_250['percents'].sum(index=1)
top_250['percents'] = top_250['percents']/sum_percents*total_sample_size


with sqlite3.connect('../exploratory_analysis/database.sqlite') as conn:
	with open('comments_data','a') as out_f:

		print 'Opened connection'

		cursor = conn.cursor()

		for i in range(len(top_250)):
			table = pd.read_sql_query("SELECT subreddit,body FROM May2015 WHERE length(body)>0 AND body <> '[deleted]' AND subreddit='"+top_250.ix[i,0] +"' LIMIT "+str(int(top_250.ix[i,1])), conn)
			
			table.to_csv(out_f, sep='\t',header=False,encoding='utf-8',index=False)


