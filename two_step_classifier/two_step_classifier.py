from clustering import *
from sklearn.tree import DecisionTreeClassifier


class two_layer_classifier(object):

	'''
	First run the cluster algorithm. 
	Use the results of clustering to generate new labels for the training data - each comment has a cluster id
	Classifier 1: predict the cluster label. Train using y_train_cluster.
	Classifier 2: predict the subreddit name. Train using y_train_subreddit.
	'''

	def __init__(self, num_clusters):

		self.num_clusters = num_clusters
		self.x_train, self.y_train_subreddit, self.y_train_cluster, self.x_test, self.y_test_subreddit, self.y_test_cluster = self.load_data()


	def load_data(self):

		#Load the training data and the corresponding subreddit labels
		x_train = pickle.load(open('../generate_train_test/X_train.p', 'rb'))
		y_train_subreddit = pickle.load(open('../generate_train_test/y_train.p', 'rb'))
		x_test = pickle.load(open('../generate_train_test/X_test.p', 'rb'))
		y_test_subreddit = pickle.load(open('../generate_train_test/y_test.p', 'rb'))

		#Run the clustering algorithm and get a list containing the cluster labels
		c = clustering(self.num_clusters)
		clusters = c.get_clusters()

		y_train_cluster = np.zeros(len(y_train_subreddit))
		y_test_cluster = np.zeros(len(y_test_subreddit))

		#Loop through each cluster 
		for i in range(c.default_k):

			#Get a Boolean array indicating whether the indices of y_subreddit contain the elements in clusters[i], then multiply by i
			#Since y_cluster is initialized to an array of zeros, and since each subreddit name is only in one cluster, y_cluster will finally 
			#containing indices indicating the cluster membership of the datapoint
			y_train_cluster += np.in1d(y_train_subreddit, clusters[i])*i
			y_test_cluster += np.in1d(y_test_subreddit, clusters[i])*i

		return x_train, y_train_subreddit, y_train_cluster, x_test, y_test_subreddit, y_test_cluster


	def first_layer_classifier(self):

		'''Train a classifier on x_train and y_train_cluster'''
		clf = DecisionTreeClassifier()
		clf = clf.fit(self.x_train, self.y_train_cluster)
		pred = clf.predict(self.x_test)
		return clf.predict(self.x_test)


	def accuracy(self, labels, preds):

		return sum(np.equal(labels,pred))/float(len(pred))


	def second_layer_classifier(self):

		#


def main():

	c = two_layer_classifier(5)
	test_pred = c.first_layer_classifier()

	print accuracy(y_test_cluster, test_pred)

if __name__ == "__main__":
    main()
