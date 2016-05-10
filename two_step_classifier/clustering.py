from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import sys

'''
Resources: 
TF-IDF: http://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html
Clustering: http://scikit-learn.org/stable/modules/clustering.html

Clustering Algorithm:

1. K-means scales relatively well and there is also the option to use small-batch k-means. 
However it responds poorly to clusters that are not convex and not isotropic. 

2. Affinity propogation chooses the number of clusters for you. However, it doesn't appear to scale
well for large datasets. The feasibility of this algorithm will be dependent on the size of the training set.

3. Mean shift: not scalable. 
'''


class clustering(object):

	'''
	Creates a tool for clustering subreddits. Also has utility to: describe the top N tokens by tf-idf for a subreddit,
	describe clusters contents, compute and plot dispersion as a function of number of clusters.
	'''


	def __init__(self, k, top_n_value=100):

		self.default_k = k
		self.top_n_value = top_n_value
		self.subreddits, self.docs = self.get_docs()
		self.top_n_arrays = self.top_tfidf_feature_extraction()
		self.top_n_bool = self.to_bool()
		self.pred = self.k_means(self.top_n_bool, self.subreddits, k)


	def get_docs(self):

		'''
		Load only training data. For each subreddit, create a document containing appended comments. 
		'''

		#Create a very small pickle file to test this out on. 
		x_train = pickle.load(open('../generate_train_test/final_dataset/X_train.p', 'rb'))
		y_train = pickle.load(open('../generate_train_test/final_dataset/y_train.p', 'rb'))

		#Keys are subreddits, values are strings containing appended comments data.
		corpus = {}

		for i in range(len(y_train)):

			#If we've already seen this subreddit, append to the existing string. 
			if y_train[i] in corpus:
				corpus[y_train[i]]+=(' ' + x_train[i])
			
			#Otherwise create a new key-value pair. 
			else:
				corpus[y_train[i]]=x_train[i]

		#Create lists of subreddit names and corresponding documents - works easy with sklearn
		subreddits = []
		docs = []
		for subreddit in corpus:
			subreddits.append(subreddit)
			docs.append(corpus[subreddit])

		return np.array(subreddits), docs


	def top_tfidf_feature_extraction(self):

		'''Get the top n tokens by TF-IDF for each subreddit document; to be used as the vector representation of each subreddit.
		Resource: http://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score'''

		tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1, smooth_idf=True)
		docs_tfidf = tfidf_vectorizer.fit_transform(self.docs)

		#This array indicates the vocabulary and indices that correspond to each word in the corpus's dictionary
		lexical_dict = np.array(tfidf_vectorizer.get_feature_names())

		top_n_arrays = []

		for i in range(len(self.docs)):
				
			#Sort the TF-IDF results (descending) for each document by value
			tfidf_sorting = np.argsort(docs_tfidf[i].toarray()).flatten()[::-1]
			#Use the top n tokens as a vector representation of the current subbredit
			top_n_arrays.append(lexical_dict[tfidf_sorting][:self.top_n_value])

		return top_n_arrays


	def to_bool(self):

		'''Each subreddit has its own list of top_n words. Cull these into an ordered set, 
		and re-define the representative vector for each subreddit as a boolean vector corresponding to the indices in the ordered set.
		It should be True in indices that correspond to words that are in the top n for that subreddit vector, and False otherwise.
		Resource: http://docs.scipy.org/doc/numpy/reference/generated/numpy.in1d.html#numpy.in1d
		'''

		#Concatenate all the top n arrays and drop duplicate words
		top_n_dict = np.unique(np.concatenate(self.top_n_arrays, axis = 0))

		#For each subreddit, in1d creates an array that is False where the value in top_n_dict is not in the top_n array
		#And is True where the value in top_n_dict *is* in the top_n array
		top_n_bool = np.array([np.in1d(top_n_dict, self.top_n_arrays[i]) for i in range(len(self.top_n_arrays))])

		return top_n_bool


	def k_means(self, X, y, k=None):

		if k:
			kmeans = KMeans(n_clusters = k)
		else:
			kmeans = KMeans(n_clusters = self.default_k)

		return kmeans.fit_predict(X,y)


	def compute_dispersion(self, pred=None, k=None):

		'''
		Compute the average distance between elements in a cluster. 
		Entropy is not a viable measure since subreddits are unique - each cluster necessarily has entropy equal to 1.

		Option 1: compute pairwise distance of every element in the cluster, then take the average. This has complexity O(n^2).
		Option 2: compute the element-wise average or "center" of the cluster. Compute n pairwise distances, and take their average. 
		This has complexity O(n). Since we want to build an algorithm that scales for a large number of classes, Option 2 is better.
		
		This function is variable for different values of pred and k so that the default will compute average dispersion 
		measures for the instantiating k value 
		'''

		if pred is None:
			pred = self.pred
		if not k:
			k = self.default_k

		avgs = []
		stds = []
		cluster_sizes = []

		#Loop through each cluster
		for i in range(k):

			#Get an array of the boolean representations of the subreddits which are in this cluster
			cluster = self.top_n_bool[np.where(pred == i)]

			#Compute the geometric center of the cluster
			center = np.mean(cluster, axis = 0)

			#Compute the L2 distances of each vector/subreddit from the center
			dist = []
			for j in range(len(cluster)):
				dist.append(np.linalg.norm(center-cluster[j]))

			#Save the cluster sizes, the average distance and the standard deviation of the distances
			cluster_sizes.append(len(cluster))
			avgs.append(np.mean(dist))
			stds.append(np.std(dist))

		#Finally, report the weighted mean of the means and the weighted mean of the standard deviations
		#Take the weighted mean so that clusters of size 1 (or small size) do not unfairly drag down the value
		return np.average(avgs, weights = cluster_sizes), np.average(stds, weights = cluster_sizes)


	def plot_dispersion(self, cap):

		'''
		Plot dispersion measure as a function of number of clusters.
		'''
		
		means = []
		stds = []

		for i in range(cap):
			pred = self.k_means(self.top_n_bool, self.subreddits, i+1)
			c_mean, c_std = self.compute_dispersion(pred, i+1)
			means.append(c_mean)
			stds.append(c_std)

		plt.errorbar(np.arange(cap)+1, means, stds)
		plt.show()


	def get_clusters(self, pred=None, num_clusters = None):

		'''
		This function is flexible for different prediction vectors and values for num_clusters 
		so that you can attempt clustering algorithms with different values for n without re-loading
		and computing top_n_bool, etc. 
		'''

		if not num_clusters:
			num_clusters = self.default_k
		if not pred:
			pred = self.pred

		clusters = []
		for i in range(num_clusters):
			clusters.append(self.subreddits[np.where(pred == i)])

		#The list at index i of clusters is the list of subreddit names in cluster i
		return clusters


def main():

	c = clustering(5)
	c.plot_dispersion(10)
	print c.get_clusters()


if __name__ == "__main__":
    main()







