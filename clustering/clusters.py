from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import sys

'''
Resources: http://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html
'''


def feature_extraction():

	'''
	Load only training data. For each subreddit, create a document of appended comments. 
	Compute TF-IDF scores. Dropping stop-words, pick 150 words with the highest TF-IDF score. 
	'''

	x_train = pickle.load(open('../generate_train_test/X_train.p', 'rb'))
	y_train = pickle.load(open('../generate_train_test/y_train.p', 'rb'))

	#Keys are subreddits, values are strings containing appended comments data.
	corpus = {}

	j = 0
	for i in range(len(y_train)):

		#If we've already seen this subreddit, append to the existing string. 
		if y_train[i] in corpus:
			corpus[y_train[i]]+=comment
		
		#Otherwise create a new key-value pair. 
		else:
			corpus[y_train[i]]=comment
	

def find_clusters():

	'''
	Use cosine similarity to determine clusters. 
	'''
	pass

def compute_dispersion():

	'''
	For now, use entropy. 
	'''
	pass


feature_extraction()