from importlib import import_module
import bottleneck as bn
from clustering import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.core import XGBoostError
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
import sys
import time

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

	def __init__(self, cluster_method, cluster_params, num_suggestions, x_train_file, x_test_file, y_train_file, y_test_file):

		#Simplify training/development process...
		self.x_train_file = x_train_file
		self.x_test_file = x_test_file
		self.y_train_file = y_train_file
		self.y_test_file = y_test_file

		#For each point, emit a list of ranked predictions
		self.num_suggestions = num_suggestions

		if cluster_method == 'manual':
			self.num_clusters = len(cluster_params)
			self.clusters = cluster_params

		else:
			#Clustering tools and information
			self.num_clusters = cluster_params[0]
			#Generate clusters immediately and store the tool for later use
			self.c = clustering(cluster_params[0], cluster_params[1])
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
		x_train_raw = pickle.load(open(self.x_train_file, 'rb'))
		x_test_raw = pickle.load(open(self.x_test_file, 'rb'))

		tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1, smooth_idf=True).fit(x_train_raw)
		x_train = tfidf_vectorizer.transform(x_train_raw)
		x_test = tfidf_vectorizer.transform(x_test_raw)

		return x_train, x_test, x_train_raw, x_test_raw


	def load_y_data(self):

		#Load the subreddit labels
		y_train_subreddit = pickle.load(open(self.y_train_file, 'rb'))
		y_test_subreddit = pickle.load(open(self.y_test_file, 'rb'))

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

		clf = LogisticRegression(penalty='l2',multi_class='ovr',C=3.16)
		clf = clf.fit(self.x_train, self.y_train_cluster)
		if pred_on_train:
			print 'First layer accuracy: ', clf.score(self.x_train, self.y_train_cluster)
			return clf.predict_proba(self.x_train)
		else:
			print 'First layer accuracy: ', clf.score(self.x_test, self.y_test_cluster)
			return clf.predict_proba(self.x_test)


	def train_second_layer_classifiers(self):

		models = []

		for i in range(self.num_clusters):
			c_x_train = self.x_train[np.where(self.y_train_cluster == i)]
			c_y_train_subreddit = self.y_train_subreddit[np.where(self.y_train_cluster == i)]

			try:
				clf = AdaBoostClassifier(base_estimator=SGDClassifier(loss="log"), n_estimators=100, learning_rate=0.5,)
				models.append(clf.fit(c_x_train, c_y_train_subreddit))

			except (ValueError,XGBoostError):
				#Some sklearn models won't train the model if the training set has only one class - but Decision Trees will
				#Using sklearn's Decision Tree tool allows us to conveniently use sklearn predict_proba and classes_ functions below
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

		ct = 0
		for i in range(len(labels)):
			if np.in1d(labels[i],preds[i]):
				ct += 1

		return ct/float(len(labels))


	def first_layer_hard_accuracy(self,pred_on_train):

		'''Might be useful to know how well the first layer classifer performs for 0-1 loss'''

		clf = DecisionTreeClassifier()
		clf = clf.fit(self.x_train, self.y_train_cluster)
		if pred_on_train:
			return sum(np.equal(self.y_train_cluster,clf.predict(self.x_train)))/float(len(self.y_train_cluster))
		else:
			return sum(np.equal(self.y_test_cluster,clf.predict(self.x_test)))/float(len(self.y_test_cluster))


def main():

	x_train = '../generate_train_test/final_dataset/X_train.p'
	x_validate = '../generate_train_test/final_dataset/X_validate.p'
	x_test = '../generate_train_test/final_dataset/X_test.p'
	y_train = '../generate_train_test/final_dataset/y_train.p'
	y_validate = '../generate_train_test/final_dataset/y_validate.p'
	y_test = '../generate_train_test/final_dataset/y_test.p'

	results_filename = '../results/ada_lr.txt'

	#num_clusters = [5,8,10,12,15]
	num_clusters = [10]
	top_tfidf = [250]

	with open(results_filename,'wb') as out_f:

		for num_tfidf in top_tfidf: 
			for num_c in num_clusters:

				out_f.write(str(num_c)+','+str(num_tfidf)+'\n') 
				print str(num_c)+','+str(num_tfidf)
				start_time = time.time()

				#Usage: cluster_params = [num_cluster, top_n_tfidf]
				c = soft_two_layer_classifier(cluster_method='new', cluster_params=[num_c,num_tfidf], num_suggestions=5,x_train_file=x_train, x_test_file=x_validate, y_train_file=y_train, y_test_file=y_validate)
				
				train_preds = c.classify(pred_on_train=True)
				out_f.write("Top-k precision on train set: " + str(c.top_n_accuracy(c.y_train_subreddit,train_preds))+'\n')

				test_preds = c.classify(pred_on_train=False)
				out_f.write("Top-k precision on test set: " + str(c.top_n_accuracy(c.y_test_subreddit,test_preds))+'\n')

				time_elapsed = time.time() - start_time

				out_f.write("Total time: " + str(time_elapsed/60) + " minutes\n\n")
				print "Total time: " + str(time_elapsed/60) + " minutes\n\n"

	#Usage: if cluster_method='manual', then in cluster_params, pass the list of lists containing the actual clusters: one list per cluster, containing subreddit names
	#cluster_example = [['funny', 'todayilearned', 'nfl', 'pics', 'news'], ['videos'], ['AskReddit'], ['leagueoflegends'], ['pcmasterrace']]
	#c2 = soft_two_layer_classifier(cluster_method='manual', cluster_params=cluster_example, num_suggestions=5,x_train_file=x_train, x_test_file=x_test, y_train_file=y_train, y_test_file=y_test)
	


if __name__ == "__main__":
    main()
