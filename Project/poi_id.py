#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")
# These functions were provided by Udacity
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") ) 

print('####### Data Exploration #########')
people = data_dict.keys()
print('Total number of data points: %d' % len(people))
POI_count = 0
for person in people:
    POI_count += data_dict[person]['poi']    
print('Number of flagged Persons of Interest: %d' % POI_count)
print('Number of people without POI flag: %d' % (len(people) - POI_count))


### Feature Exploration
all_features = data_dict['CORDES WILLIAM R'].keys()
print('Each person has %d features available' %  len(all_features))
### Evaluate dataset for completeness
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in people:
    records = 0
    for feature in all_features:
        if data_dict[person][feature] == 'NaN':
            missing_values[feature] += 1
        else:
            records += 1

### Print results of completeness analysis
print('Number of Missing Values for Each Feature:')
for feature in all_features:
    print("%s: %d" % (feature, missing_values[feature]))

### Is anyone missing all financial information?
incomplete_money = []
for person in data_dict.keys():  
    if data_dict[person]['total_payments'] == 'NaN' and \
    data_dict[person]['total_stock_value'] == 'NaN':
        incomplete_money.append(person)
print
if len(incomplete_money) > 0:
    print('The following people have no data for payments and stock value:')
    counter = 0
    for person in incomplete_money:
        print person        
        counter += data_dict[person]['poi']
    print('Of these %d people %d of them are POIs' % (len(incomplete_money), 
          counter))
else:
    print('No person is missing both payments and stock value.')
print
    
### Is anyone missing all email information?
incomplete_email = []
for person in data_dict.keys():
    if data_dict[person]['to_messages'] == 'NaN' and \
       data_dict[person]['from_messages'] == 'NaN':
        incomplete_email.append(person)    
if len(incomplete_email) > 0:
    print('The following people have no message data for emails:')
    counter = 0
    for person in incomplete_email:
        print("%s, POI: %r" % (person, data_dict[person]['poi']))        
        counter += data_dict[person]['poi']
    print('Of these %d people, %d of them are POIs' % (len(incomplete_email), counter))
else:
    print('No person is missing both to and from message records.')

############# Task 2: Remove outliers #######################
def OutlierPlot(data_dict, xFeature, yFeature, flag='poi'):
    'Create a scatterplot to identify outliers. Use first feature to flag data.'
    
    data = featureFormat(data_dict, [flag, xFeature, yFeature])  
    ### Plot features with flag=True in red
    ### plotting code copied from Udacity course data
    import matplotlib.pyplot as plt
    for point in data:
        x = point[1]
        y = point[2]
        if point[0]:
            plt.scatter(x, y, color="r", marker="*")
        else:
            plt.scatter(x, y, color='b', marker=".")
    plt.xlabel(xFeature)
    plt.ylabel(yFeature)
#    picture = xFeature + yFeature + '.png'
#    plt.savefig(picture, transparent=True)
    plt.show()

### Check for outliers between financial features
OutlierPlot(data_dict, 'total_payments', 'total_stock_value')
print('The obvious outlier belongs to TOTAL, so it is removed')
data_dict.pop( 'TOTAL', 0 ) # remove spreadsheet total line
#import os
#os.rename('total_paymentstotal_stock_value.png', 
#          'total_paymentstotal_stock_valuebefore.png')
OutlierPlot(data_dict, 'total_payments', 'total_stock_value')
print('The obvious outlier point belongs to Ken Lay, so it is left in.')

### Explore email features
OutlierPlot(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
print 'Email messages with POIs has a large range of values.'
OutlierPlot(data_dict, 'from_messages', 'to_messages')
print 'Total emails also have a large range.'  

print
print('############### Feature Selection ###################')
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". Start with all features except email addr.
features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', \
                 'deferred_income', 'deferral_payments', 'loan_advances', \
                 'other', 'expenses', 'director_fees', 'restricted_stock', \
                 'exercised_stock_options', 'restricted_stock_deferred', \
                 'total_payments', 'total_stock_value', \
                 'from_poi_to_this_person', 'from_this_person_to_poi', \
                 'shared_receipt_with_poi', 'from_messages', 'to_messages']

def FeatureSelection(data_dict, features_list):                
    # Convert dictionary to numpy array, converts NaN to 0.0                  
    data = featureFormat(data_dict, features_list, \
                         sort_keys = True, remove_all_zeroes = False)
    # Separate into labels = 'poi' and features = rest of features_list
    labels, features = targetFeatureSplit(data)
    
    from sklearn.feature_selection import RFECV 
    # Recursive Feature Elimination with Cross Validation
    from sklearn.svm import SVC
    # Support Vector Classifier to estimate fit coefficients for each feature
    from sklearn.cross_validation import StratifiedShuffleSplit
    # cross validation maintain roughly equal number of POIs in each split
    
    ### Create Estimator 
    # which will update the coefficients with each iteration
    # class weight is set to auto because of unbalanced data classes
    # weight will be inversely proportional to class size
    svc = SVC(kernel='linear', class_weight='auto', random_state=42)
    ############## Scale features ######################
    # SVC algorithm requires use scaled features
    # missing values are coded 0.0, so MinMax will preserve those zero values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    ### Select cross-validation method
    # StratifiedShuffleSplit keeps roughly the same number of POIs in each split 
    sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=42)
    ### Select evaluation metric
    # Evaluate model using f1 = 2 * (precision * recall) / (precision + recall)
    # Model should be able to predict POIs, which are a small percentage of cases
    metric = 'f1'
    # run the feature eliminater
    rfecv = RFECV(estimator=svc, cv=sss, scoring=metric, step=1)
    rfecv = rfecv.fit(features, labels)
    
    # view results
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score using F1 (precision&recall)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#    plt.savefig('featureSelection.png', transparent=True)
    plt.show()
    print("Optimal number of features is %d" % rfecv.n_features_)
    print('Features selected by recursive feature elimination with cross validation:')
    F1_score = round(rfecv.grid_scores_[rfecv.n_features_], 3)
    print('F1 score from optimal features: %r' % F1_score)
    selection = rfecv.get_support()
    selected_features = ['poi']
    rejected_features = []
    for i in range(len(selection)):
        if selection[i]:
            selected_features.append(features_list[i + 1]) # first feature is 'poi'=the label
        else:
            rejected_features.append(features_list[i + 1])
    print(selected_features[1:])
    print('Features eliminated:')
    print(rejected_features)
    return selected_features, F1_score
    
features_list, initial_F1 = FeatureSelection(data_dict, features_list)

####################################################################
### Task 3: Create new feature(s)
### Give emails to/from pois as a proportion of total emails
### Create new features as keys in data_dict
for person in data_dict.keys():
    to_poi = float(data_dict[person]['from_this_person_to_poi'])
    to_all = float(data_dict[person]['from_messages'])
    if to_all > 0:
        data_dict[person]['fraction_to_poi'] = to_poi / to_all
    else:
        data_dict[person]['fraction_to_poi'] = 0
    from_poi = float(data_dict[person]['from_poi_to_this_person'])
    from_all = float(data_dict[person]['to_messages'])
    if from_all > 0:
        data_dict[person]['fraction_from_poi'] = from_poi / from_all
    else:
        data_dict[person]['fraction_from_poi'] = 0

############# Evaluate new features
### Add new feature to features_list
features_list.extend(['fraction_to_poi', 'fraction_from_poi'])

features_list, second_F1 = FeatureSelection(data_dict, features_list)

#### keep the engineered features added to data_dict
my_dataset = data_dict

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='auto', random_state=42)
from time import time
from tester import test_classifier
t0 = time()
test_classifier(rf, data_dict, features_list, folds = 100)
print("Random forest fitting time: %rs" % round(time()-t0, 3))

###### Adaboost
from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier(random_state=42)
t0 = time()
test_classifier(ab, data_dict, features_list, folds = 100)
print("AdaBoost fitting time: %rs" % round(time()-t0, 3))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Convert dictionary to numpy array, converts NaN to 0.0                  
data = featureFormat(data_dict, features_list, \
                     sort_keys = True, remove_all_zeroes = False)
# Separate into labels = 'poi' and features = rest of features_list
labels, features = targetFeatureSplit(data)

from sklearn.grid_search import GridSearchCV
rf_params = {'max_features': range(1,5), 'n_estimators': range(10, 101, 10)}
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=42)
metric = 'f1'
# use same estimator (rf, random forest) as above; add suffix t for tuned
t0 = time()
rft = GridSearchCV(rf, rf_params, scoring=metric, cv=sss)
print("Random Forest tuning: %r" % round(time()-t0, 3))
t0 = time()
rft = rft.fit(features, labels)
print("Random forest fitting time: %rs" % round(time()-t0, 3))
rf = rft.best_estimator_
t0 = time()
test_classifier(rf, data_dict, features_list, folds = 100)
print("Random Forest evaluation time: %rs" % round(time()-t0, 3))

from sklearn.tree import DecisionTreeClassifier
dt = []
for i in range(5):
    dt.append(DecisionTreeClassifier(max_depth=(i+1)))
ab_params = {'base_estimator': dt, 'n_estimators': range(50, 101, 10)}
t0 = time()
abt = GridSearchCV(ab, ab_params, scoring=metric, cv=sss)
print("AdaBoost tuning: %r" % round(time()-t0, 3))
t0 = time()
abt = abt.fit(features, labels)
print("AdaBoost fitting time: %rs" % round(time()-t0, 3))
ab = abt.best_estimator_
t0 = time()
test_classifier(ab, data_dict, features_list, folds = 100)
print("AdaBoost evaluation time: %rs" % round(time()-t0, 3))

### Select tuned adaboost as best classifier
clf = ab
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)