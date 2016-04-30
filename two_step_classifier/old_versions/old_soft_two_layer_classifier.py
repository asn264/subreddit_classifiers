from clustering import *
from sklearn.tree import DecisionTreeClassifier


class soft_two_layer_classifier(object):

	'''
	First run the cluster algorithm. 
	Use the results of clustering to generate new labels for the training data - each comment has a cluster id
	Classifier 1: predict the cluster label. Train using y_train_cluster.
	Classifier 2: predict the subreddit name. Train using y_train_subreddit.
	Output a list of suggestions/predictions for each point. Measure performance by whether this list contains the true label. 
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


	def second_layer_classifier(self, ongoing_preds, c_cluster_preds, pred_on_train):

		'''Now each cluster only contains a few subreddits. For each cluster, train a model on x_train data within that cluster,
		and predict a subreddit name (within the cluster).'''

		second_preds = []
		
		for i in range(self.num_clusters):

			c_clf = self.second_stage_models[i]

			#Get a list of all the training/testing points predicted to be in this cluster
			mask = (c_cluster_preds == i) & np.any(ongoing_preds=='0', axis=1)
			if pred_on_train:
				c_x_to_pred = self.x_train[mask]
			else:
				c_x_to_pred = self.x_test[mask]

			#Now predict the subreddit id for all of these training/testing points 
			try:
				new_preds = c_clf.predict_proba(c_x_to_pred)
				new_preds_names = c_clf.classes_[np.argsort(-new_preds)]

				#Get the rankings to tell you the most likely class
				print ongoing_preds

				curr_pred = 0
				for i in range(len(mask)):
					if mask[i]:
						#print ongoing_preds[i]
						ongoing_preds[i] = new_preds_names[curr_pred]
						print new_preds_names[curr_pred]
						#print ongoing_preds[i]
						curr_pred += 1

				#Glue it onto ongoing preds appropriately

				#print self.clusters[i]

			#If nothing was predicted to be in the current cluster, then c_x_test is empty and a ValueError will be raised
			except ValueError:
				pass

		return second_preds



	def classify(self, pred_on_train):

		first_preds = self.first_layer_classifier(pred_on_train)

		#Each row is a ranking of the top n predicted clusters for that element. Give n = num_suggestions.
		cluster_preds = (-first_preds).argsort(axis=1)[:,0:self.num_suggestions]
				
		#So as long as these arrays still contain the string 'still_needs_pred', we have to generate more second_preds
		if pred_on_train:
			final_preds = np.chararray((len(self.x_train_raw),self.num_suggestions), itemsize=50)
		else:
			final_preds = np.chararray((len(self.x_test_raw),self.num_suggestions), itemsize=50)
		final_preds[:] = '0'

		i = 0
		#In each iteration you are looking at the ith most likely cluster prediction
		while np.sum(np.ravel(final_preds)=='0') > 0:
			self.second_layer_classifier(final_preds, cluster_preds[:,i], pred_on_train)
			i += 1
			sys.exit()



	def top_n_accuracy(self, labels, preds):

		'''For each point, consider a list of size n, and consider the prediction correct if the list contains the correct label anywhere.'''

		pass


def main():

	c = soft_two_layer_classifier(num_suggestions=2, num_clusters=5)
	c.classify(pred_on_train=True)


if __name__ == "__main__":
    main()
