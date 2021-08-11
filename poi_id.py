#!/usr/bin/python

import sys
import pickle

from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
import numpy as np

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Data Exploration and Removing outliers
print 'Data Exploration and Removing Outliers'

# main data exploration function
def poi_missing_email_data():
    # Find total count and values of POI with missing or no to/from email information
    poi_count = 0
    poi_keys = []
    for person in data_dict.keys():
        if data_dict[person]["poi"]:
            poi_count += 1
            poi_keys.append(person)

    poi_missing_emails = []
    for poi in poi_keys:
        if (data_dict[poi]['to_messages'] == 'NaN' and data_dict[poi]['from_messages'] == 'NaN') or \
            (data_dict[poi]['to_messages'] == 0 and data_dict[poi]['from_messages'] == 0):
            poi_missing_emails.append(poi)

    return poi_count, poi_missing_emails

people_val = data_dict.keys()
feature_val = data_dict[people_val[0]]
poi_count_n, poi_missing_emails = poi_missing_email_data()

print 'The number of people in the dataset is: %d' % len(people_val)
print 'The number of features for each person: %d' % len(feature_val)
print 'Number of Persons of Interests (POIs) in dataset: %d out of 34 total POIs' % poi_count_n
print 'Number of non-POIs in dataset: %d' % (len(people_val) - poi_count_n)
print 'POIs with zero or missing to/from email messages in dataset: %d' % len(poi_missing_emails)
print poi_missing_emails

### Removing Outliers
#finding outliers in salary to bonus ratios

#removing outlier
#Only outlier found was the Total field which has no relevance to this study
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

### Task 3: Create new feature(s)
#looking at high earning individuals and their bonus could be a valid way of identifying POI's
#an unusually high bonus compared to salary could show signs of embezzlement etc hence a salary vs bonus
#ratio is examined
def create_bonus_feat():
    people_val=data_dict.keys()
    for person in people_val:
        personal_salary = float(data_dict[person]['salary'])
        personal_bonus = float(data_dict[person]['bonus'])
        if personal_bonus > 0 and personal_salary > 0:
            data_dict[person]['bonus_salary_ratio']=data_dict[person]['salary']/data_dict[person]['bonus']
        else:
            data_dict[person]['bonus_salary_ratio']=0
    features_list.extend(['bonus_salary_ratio'])
    return
create_bonus_feat()

print features_list
### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#scaling
from sklearn.preprocessing import MinMaxScaler
import numpy

scaler = MinMaxScaler()
scaler.fit_transform(features)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=42)

#feature selection using SelectKbest
 #Extract features and labels from dataset for local testing


skbest = SelectKBest(k=4)  # try best value to fit
sk_transform = skbest.fit_transform(features_train, labels_train)
indices = skbest.get_support(True)
print skbest.scores_

n_list = ['poi']
for index in indices:
    print 'features: %s score: %f' % (features_list[index + 1], skbest.scores_[index])
    n_list.append(features_list[index + 1])

print n_list

#redefining features_list as to contain the variables found using kbest
features_list = n_list


### Task 4: Try a variety of classifiers


## Decision Tree Grid Search ##
from sklearn.tree import DecisionTreeClassifier

print("Fitting the classifier to the training set")
t0 = time()

param_tree = {'criterion':['gini', 'entropy'],
              'splitter':['best','random'],
              'min_samples_split':[2,5,10,15]
             }
clf_tree = GridSearchCV(DecisionTreeClassifier(), param_tree,scoring='recall',cv=2)
clf_tree = clf_tree.fit(features_train, labels_train)

print("done in %0.3fs" % (time() - t0))
print clf_tree.best_estimator_


#Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#Decision Tree
clf_final = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='random')

clf = clf_final.fit(features_train, labels_train)



test_classifier(clf, my_dataset, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)