'''
Sample new dataset from sqlite database in proportion to subreddit prevalence in database. 
Make sure to drop empty comments, deleted comments, and comments with 3 words or less.
'''

import pandas as pd
import sqlite3
import pickle

#2.5 million rows
total_sample_size = 1000000


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
pickle.dump(tot_df, open("clean_df.p", "wb"))

