from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
from numpy import mean
from tester import test_classifier

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
# evaluate function inputs classifier name, parameters, dataset, featuere list
#and returns the classifier with best tuned parameters using gridsearch cv

def evaluate(clf, dataset, feature_list, features, labels, num_iter, params):
    pipeline = Pipeline(steps=[("scaler", scaler), ("skb", SelectKBest(k='all')), ("clf", clf)])

    # split features and lablels in to training and testing data using train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    # make 3 lists to store metric for each iteration
    precision_val = []
    recall_val = []
    accuracy_val = []

    
    for i in xrange(0, num_iter):
        #print params
        #perform grid search
        clf = GridSearchCV(pipeline, param_grid=params)
        clf.fit(features_train, labels_train)
        print '*****************************'
        # print out best estimator and best parameter
        print (clf, "best estimators", clf.best_estimator_)
        print (clf, "best paramaters", clf.best_params_)
        # make predictions using best estimator        
        pred = clf.predict(features_test)
        #store the metric in the list
        precision_val.append(precision_score(labels_test, pred))
        recall_val.append(recall_score(labels_test, pred))
        accuracy_val.append(accuracy_score(labels_test, pred))
    # print mean metrics
    print 'Mean Recall score: ', mean(recall_val)
    print 'PMean recision score: ', mean(precision_val)
    print 'Mean Accuracy score: ' , mean(accuracy_val)
#    print(clf)
    # return the final estimator
    return clf

