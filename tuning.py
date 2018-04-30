from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from numpy import mean
from tester import test_classifier

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def evaluate(clf, dataset, feature_list, features, labels, num_iter, params):
    
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)



    precision_val = []
    recall_val = []
    accuracy_val = []
    print clf
    for i in xrange(0, num_iter):
        #print params
        clf = GridSearchCV(clf, params)
        clf.fit(features_train, labels_train)
        print '*****************************'
        print clf.best_estimator_
        print clf.best_params_

        clf = clf.best_estimator_
        #test_classifier(clf, dataset, feature_list)
        pred = clf.predict(features_test)
        precision_val.append(precision_score(labels_test, pred))
        recall_val.append(recall_score(labels_test, pred))
        accuracy_val.append(accuracy_score(labels_test, pred))
    print 'Mean Recall score: ', mean(recall_val)
    print 'PMean recision score: ', mean(precision_val)
    print 'Mean Accuracy score: ' , mean(accuracy_val)

