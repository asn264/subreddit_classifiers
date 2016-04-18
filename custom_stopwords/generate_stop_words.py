'''
Sample new dataset from sqlite database in proportion to subreddit prevalence in database. 
Make sure to drop empty comments and deleted comments as well as comments that are not in English characters. 
Do word counts using Collections class.
'''

from collections import Counter
import pandas as pd
import sqlite3
import pickle
import sys


def generate_counts():

	'''
	Sample from dataset in proportion to subreddit counts. Then take word counts.
	'''

	total_sample_size = 10**6

	subreddit_data = pd.read_csv('../exploratory_analysis/subreddit_count.tsv',delimiter='\t')
	top_250 = subreddit_data[:250][[1,3]]
	sum_percents = top_250['percents'].sum(index=1)
	top_250['percents'] = top_250['percents']/sum_percents*total_sample_size


	#Start with empty list
	global_li = []
	with sqlite3.connect('../exploratory_analysis/database.sqlite') as conn:

		print 'Opened connection'
		cursor = conn.cursor()

		for i in range(len(top_250)):

			print 'Querying subreddit ', i

			table = pd.read_sql_query("SELECT subreddit,body FROM May2015 WHERE length(body)>0 AND body <> '[deleted]' AND subreddit='"+top_250.ix[i,0] +"' LIMIT "+str(int(top_250.ix[i,1])), conn)

			#Loop through rows of table and extract comment
			for i in range(len(table)):

				#Append words to list
				global_li.extend(table['body'][i].strip().split())

	pickle.dump(Counter(global_li), open("counts_"+str(total_sample_size)+".p", "wb"))

def get_top_n():

	'''
	Open pickle file and get top N words by count.
	'''

	counts = pickle.load(open('counts_1000000.p', 'rb'))
	print counts.most_common(100)


