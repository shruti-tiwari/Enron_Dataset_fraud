#!/usr/bin/python

import sys
import pickle
sys.path.append("/Users/admin/Desktop/DAND/Projects/enron/final_project/tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from feature_select import select_k_best, dict_to_list
from tuning_backup import evaluate

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus', 'deferral_payments', 'deferred_income', 'director_fees',
        'exercised_stock_options', 'expenses',
       'from_messages', 'loan_advances', 'long_term_incentive',
       'other' , 'restricted_stock', 'restricted_stock_deferred',
       'salary', 'shared_receipt_with_poi', 'to_messages',
       'total_payments', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
     #removing entry for 'TOTAL' in salary
    data_dict.pop('TOTAL')
    data_dict.pop('LOCKHART EUGENE E')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    
 ### remove NAN's from dataset
    # Update NaN values with 0 except for email address
    people_keys = data_dict.keys()
    feature_keys = data_dict[people_keys[0]]
    nan_features = {}
    # Get list of NaN values and replace them
    for feature in feature_keys:
        nan_features[feature] = 0
    for person in people_keys:
        for feature in feature_keys:
            if feature != 'email_address' and \
                data_dict[person][feature] == 'NaN':
                data_dict[person][feature] = 0
                nan_features[feature] += 1


                
    outliers = []
    for feature in features_list:
        for key in data_dict:
            val = data_dict[key][feature]
            if val == 'NaN':
                continue
            outliers.append((key, int(val)))
### find five top paid executives
    outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:10])
    ### print top 5 salaries
    #print ('final_outliers',outliers_final)   
                  

    
### Task 3: Create new feature(s)
    
    ### create two lists of new features
    fraction_from_poi_email=dict_to_list(data_dict, "from_poi_to_this_person","to_messages")
    fraction_to_poi_email=dict_to_list(data_dict, "from_this_person_to_poi","from_messages")
    
### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

### selecting top 10 features by using selectkbest

### first adding the features in to feature_list
features_list.append('fraction_from_poi_email')
features_list.append('fraction_to_poi_email')
### calling select k function from feature_select

features_best = select_k_best(data_dict, features_list, 10)
print(features_best)
# as selector removed 'poi' from the list, adding poi at first place

features_best.insert(0, 'poi')
    
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_best, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
'''from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()'''

### trying decision tree classifier and 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
scaler = MinMaxScaler()

classifiers =[{'classifier': LogisticRegression(),
                'params': {  "clf__C": [ 0.5, 1, 10, 10**2, 10**3, ],
                    "clf__tol":[10**-1, 10**-4, 10**-5,],
                    "clf__class_weight":['balanced']
                    }},
               {'classifier': tree.DecisionTreeClassifier(),
                'params':
                    {
                        "clf__criterion": ["gini", "entropy"],
                        "clf__min_samples_split": [10,15,20,25]
                    }
                },
              
                {'classifier': RandomForestClassifier(),
                'params':
                    { "clf__n_estimators": [25, 50],
                    "clf__min_samples_split": [2, 3, 4],
                    "clf__criterion": ['gini', 'entropy']
                      }}
                ]

# Validate model precision, recall and F1-score
from sklearn.pipeline import Pipeline
# these code has been used to try out three algorithms. Please uncomment to check that part
'''
for c in classifiers:
    clf = c['classifier']
    params = c['params']
    print(clf)
    cv = evaluate(clf, my_dataset, features_best, features, labels, 10, params)
#    print(cv.best_params)
'''

# following
#clf = Pipeline(steps=[("scaler", scaler), ("skb", SelectKBest(k='all')), ("clf", RandomForestClassifier(criterion = 'gini', n_estimators = 60, min_samples_split = 4, min_samples_leaf=2))])

# The final classifier with optimized paramters using gridsearch and then manually
clf = Pipeline(steps=[("scaler", scaler),  ("clf", LogisticRegression(penalty = 'l1', tol = 0.01, C =0.5, class_weight = 'balanced'))])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


test_classifier(clf, my_dataset, features_best)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.



dump_classifier_and_data(clf, my_dataset, features_list)
