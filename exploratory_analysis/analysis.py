'''
Resources:
http://stackoverflow.com/questions/305378/get-list-of-tables-db-schema-dump-etc-in-sqlite-databases
http://stackoverflow.com/questions/10065051/python-pandas-and-databases-like-mysql
http://www.tutorialspoint.com/sqlite/sqlite_python.htm
'''

import pandas as pd
import sqlite3

with sqlite3.connect('database.sqlite') as conn:

	print 'Opened connection'

	cursor = conn.cursor()
	
	#Used to find the name of the table
	#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
	#print (cursor.fetchall())

	table = pd.read_sql_query("SELECT subreddit, COUNT(*) from May2015 GROUP BY subreddit ORDER BY COUNT(*) DESC", conn)
	#table = pd.read_sql_query("SELECT COUNT(*) FROM May2015 WHERE LENGTH(body)=0", conn)

	table.to_csv('subreddit_count.tsv', sep='\t')
