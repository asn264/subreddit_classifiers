import bottleneck as bn
from clustering import *
from sklearn.tree import DecisionTreeClassifier


class soft_two_layer_classifier(object):

	'''
	First run the cluster algorithm. 
	Use the results of clustering to generate new labels for the training data - each comment has a cluster id
	Classifier 1: predict distribution on the cluster label. Train using y_train_cluster.
	Classifier 2: get distribution on the subreddit names. 
	Output a list of top n suggestions. 

	Notes: need to always output top n suggestions, because the single layer classifiers do.
	Since not every cluster will have n subreddits, need to look to other subreddits. 
	Option 1: Get as many suggestions as possible from most likely cluster, then look at next best cluster, etc. till you have enough.
	Option 2: Use the probability distribution on clusters to find a probability for each subreddit as p(cluster)p(subreddit|cluster)
	Then return the top n according to the global probability for each subreddit. 

	Option 2 is more accurate but slower: always compute subreddit number of items whereas Option 1 doesn't.
	However given the accuracy hit the algorithm takes at (clustering/first layer prediction?) given our limited dataset/computational ability,
	Option 2 was a better choice.
	'''

	def __init__(self, num_suggestions, num_clusters):

		#For each point, emit a list of ranked predictions
		self.num_suggestions = num_suggestions

		#Clustering tools and information
		self.num_clusters = num_clusters
		#Generate clusters immediately and store the tool for later use
		self.c = clustering(num_clusters)
		self.clusters = self.c.get_clusters()

		#Smoothed tfidf representations of x values
		self.x_train, self.x_test, self.x_train_raw, self.x_test_raw = self.load_x_data()
		#Subreddit labels and corresponding cluster mappings
		self.y_train_subreddit, self.y_train_cluster, self.y_test_subreddit, self.y_test_cluster = self.load_y_data()

		#Record the number of distinct classes
		self.num_subreddits = np.unique(self.y_train_subreddit).size

		#Train the models required for the second-stage prediction, since these will be the same for every call of the classify fn
		self.second_stage_models = self.train_second_layer_classifiers()


	def load_x_data(self):

		#Load the training data and convert to TF-IDF
		x_train_raw = pickle.load(open('../generate_train_test/X_train.p', 'rb'))
		x_test_raw = pickle.load(open('../generate_train_test/X_test.p', 'rb'))

		tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1, smooth_idf=True).fit(x_train_raw)
		x_train = tfidf_vectorizer.transform(x_train_raw)
		x_test = tfidf_vectorizer.transform(x_test_raw)

		return x_train, x_test, x_train_raw, x_test_raw


	def load_y_data(self):

		#Load the subreddit labels
		y_train_subreddit = pickle.load(open('../generate_train_test/y_train.p', 'rb'))
		y_test_subreddit = pickle.load(open('../generate_train_test/y_test.p', 'rb'))

		#Map the subreddit labels to the appropriate cluster label
		y_train_cluster = np.zeros(len(y_train_subreddit))
		y_test_cluster = np.zeros(len(y_test_subreddit))

		#Loop through each cluster 
		for i in range(self.num_clusters):

			#Get a Boolean array indicating whether the indices of y_subreddit contain the elements in clusters[i], then multiply by i
			#Since y_cluster is initialized to an array of zeros, and since each subreddit name is only in one cluster, y_cluster will finally 
			#containing indices indicating the cluster membership of the datapoint
			y_train_cluster += np.in1d(y_train_subreddit, self.clusters[i])*i
			y_test_cluster += np.in1d(y_test_subreddit, self.clusters[i])*i

		return y_train_subreddit, y_train_cluster, y_test_subreddit, y_test_cluster


	def first_layer_classifier(self, pred_on_train):

		'''Train a classifier on x_train and y_train_cluster'''

		clf = DecisionTreeClassifier()
		clf = clf.fit(self.x_train, self.y_train_cluster)
		if pred_on_train:
			return clf.predict_proba(self.x_train)
		else:
			return clf.predict_proba(self.x_test)


	def train_second_layer_classifiers(self):

		models = []

		for i in range(self.num_clusters):
			c_x_train = self.x_train[np.where(self.y_train_cluster == i)]
			c_y_train_subreddit = self.y_train_subreddit[np.where(self.y_train_cluster == i)]
			clf = DecisionTreeClassifier()
			models.append(clf.fit(c_x_train, c_y_train_subreddit))

		return models


	def classify(self, pred_on_train):

		'''
		Look at matrix formulation in notebook.		
		'''

		final_probs = None
		final_labels = None

		#Array of arrays containing probability distribution on clusters for items in the prediction set
		cluster_preds = self.first_layer_classifier(pred_on_train)

		for c_cluster in range(self.num_clusters):

			#For each item, what is the probability of belong to c_cluster
			c_cluster_probs = cluster_preds[:,c_cluster]	
			
			#Model trained to predict between subreddits in the current cluster		
			c_clf = self.second_stage_models[c_cluster]

			#For each item, probability distribution on subreddits in the current cluster
			if pred_on_train:
				c_subreddit_probs = c_clf.predict_proba(self.x_train)
			else:
				c_subreddit_probs = c_clf.predict_proba(self.x_test)

			#Now row i in column j contains p(item_i in c_cluster)*p(item_i is subreddit_j)
			new_probs = np.multiply(c_cluster_probs, c_subreddit_probs.T).T

			#Store the probability distribution for subreddits in the current cluster 
			if final_probs is None or final_labels is None:
				#These are always both none or neither none
				final_probs = new_probs
				final_labels = c_clf.classes_
			else:
				final_probs = np.hstack((final_probs, new_probs))
				final_labels = np.hstack((final_labels, c_clf.classes_))

		return self.get_top_n(final_probs,final_labels)


	def get_top_n(self, probs, labels):

		'''
		Get top n most likely subreddits. Each row in probs should correspond to the subreddit probability distribution 
		over an item in the prediction set. Labels should be subreddit labels which reflect the ordering of probs.
		
		Resources: http://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array
		'''

		return labels[bn.argpartsort(-probs,self.num_suggestions,axis=1)[:,:self.num_suggestions]]


	def top_n_accuracy(self, labels, preds):

		'''For each point, consider a list of size n, and consider the prediction correct if the list contains the correct label anywhere.'''

		return sum(np.in1d(labels,preds))/float(len(labels))


def main():

	c = soft_two_layer_classifier(num_suggestions=5, num_clusters=5)
	x = c.classify(pred_on_train=True)
	print c.top_n_accuracy(c.y_train_subreddit,x)

if __name__ == "__main__":
    main()
