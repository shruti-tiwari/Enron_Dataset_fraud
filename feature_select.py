#!/usr/bin/python

from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from numpy import mean
from tester import test_classifier

import sys

print(sys.path)

sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit

def select_k_best(data_dict, feature_list, num_features):
    """
    Function for selecting our KBest features.
    :param data_dict: List of employees and features
    :param feature_list: List of features to select
    :param num_features: Number (k) of features to select in the algorithm (k = 11)
    :return: Returns a list of the KBest feature names
    """
    data = featureFormat(data_dict, feature_list)
    target, features = targetFeatureSplit(data)

    clf = SelectKBest(k = num_features)
    clf = clf.fit(features, target)
    feature_weights = {}
    for idx, feature in enumerate(clf.scores_):
        feature_weights[feature_list[1:][idx]] = feature
    best_features = sorted(feature_weights.items(), key = lambda k: k[1], reverse = True)[:num_features]
    print (best_features)
    new_features = []
    for k, v in best_features:
        new_features.append(k)
    return new_features

##helperfunction for creating new features
def dict_to_list(data_dict, key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN" or data_dict[i][normalizer]== 0:
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

    
