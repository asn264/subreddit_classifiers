from clustering import *
from sklearn.tree import DecisionTreeClassifier


class hard_two_layer_classifier(object):

	'''
	First run the cluster algorithm. 
	Use the results of clustering to generate new labels for the training data - each comment has a cluster id
	Classifier 1: predict the cluster label. Train using y_train_cluster.
	Classifier 2: predict the subreddit name. Train using y_train_subreddit.
	'''

	def __init__(self, num_clusters):

		#Clustering tools and information
		self.num_clusters = num_clusters
		#Generate clusters immediately and store the tool for later use
		self.c = clustering(num_clusters)
		self.clusters = self.c.get_clusters()

		#Smoothed tfidf representations of x values
		self.x_train, self.x_test, self.x_train_raw, self.x_test_raw = self.load_x_data()
		#Subreddit labels and corresponding cluster mappings
		self.y_train_subreddit, self.y_train_cluster, self.y_test_subreddit, self.y_test_cluster = self.load_y_data()


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
			return clf.predict(self.x_train)
		else:
			return clf.predict(self.x_test)


	def second_layer_classifier(self, first_preds, pred_on_train):

		'''Now each cluster only contains a few subreddits. For each cluster, train a model on x_train data within that cluster,
		and predict a subreddit name (within the cluster).'''

		clf = DecisionTreeClassifier()

		second_preds = []
		second_labels = []

		for i in range(self.num_clusters):

			#Get a list of all the training points in this cluster 
			#y_train_cluster already tells you which cluster each training point is in.
			c_x_train = self.x_train[np.where(self.y_train_cluster == i)]
			c_y_train_subreddit = self.y_train_subreddit[np.where(self.y_train_cluster == i)]

			#Train a model on these points with corresponding y_train_subreddit labels
			c_clf = clf.fit(c_x_train, c_y_train_subreddit)

			#Get a list of all the training/testing points predicted to be in this cluster
			if pred_on_train:
				#When you evaluate on training data, still have to see which points landed in the current cluster
				#And also what subreddit labels correspond to those points
				c_x_to_pred = self.x_train[np.where(first_preds == i)]
				c_y_of_pred = self.y_train_subreddit[np.where(first_preds == i)]
			else:
				c_x_to_pred = self.x_test[np.where(first_preds == i)]
				c_y_of_pred = self.y_test_subreddit[np.where(first_preds == i)]

			#Now predict the subreddit id for all of these training/testing points 
			try:
				second_preds.append(c_clf.predict(c_x_to_pred))
				second_labels.append(c_y_of_pred)

			#If nothing was predicted to be in the current cluster, then c_x_test is empty and a ValueError will be raised
			except ValueError:
				pass

		return second_preds, second_labels


	def classify(self, pred_on_train):

		first_preds = self.first_layer_classifier(pred_on_train)
		second_preds, second_labels = self.second_layer_classifier(first_preds, pred_on_train)

		#Return both levels of prediction, as well as corresponding labels for second_preds bc of reindexing
		return first_preds, second_preds, second_labels


	def second_layer_accuracy(self, labels, preds):

		'''Expects preds and labels to be lists of lists, partitioned by predicted cluster.'''

		tot = 0
		ct = 0
		for i in range(len(preds)):
			tot += sum(np.equal(preds[i], labels[i]))
			ct += len(preds[i])
		return tot/float(ct)


	def first_layer_accuracy(self, labels, preds):

		return sum(np.equal(labels,preds))/float(len(preds))


def main():

	c = hard_two_layer_classifier(5)
	first_preds, second_preds, second_labels = c.classify(pred_on_train=False)
	print c.second_layer_accuracy(second_labels, second_preds)


if __name__ == "__main__":
    main()
