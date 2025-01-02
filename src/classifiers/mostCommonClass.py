import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

"""
code to run in main:

most_common_clf = MostCommonClassifier()
most_common_clf.fit(X_train, y_train)
y_pred = most_common_clf.predict(X_test)

"""

class MostCommonClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.most_common_class_ = None

    def fit(self, X, y):
        # Ensure inputs are numpy arrays or pandas objects
        X, y = check_X_y(X, y)
        # Identify the most frequent class using numpy
        values, counts = np.unique(y, return_counts=True)
        self.most_common_class_ = values[np.argmax(counts)]
        return self

    def predict(self, X):
        # Check the input array
        X = check_array(X)
        # Predict the most common class for all instances
        return np.full(len(X), self.most_common_class_)


    
