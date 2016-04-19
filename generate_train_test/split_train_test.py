import pandas as pd
from sklearn.cross_validation import train_test_split
import pickle

data = pickle.load( open("clean_df.p",'rb'))

X = data.values[:,1]
y = data.values[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pickle.dump(X_train, open("X_train.p", "wb"))
pickle.dump(X_test, open("X_test.p", "wb"))
pickle.dump(y_train, open("y_train.p", "wb"))
pickle.dump(y_test, open("y_test.p", "wb"))