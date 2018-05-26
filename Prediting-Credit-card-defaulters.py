# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:53:27 2018

@author: dines
"""
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as pt
from sklearn.model_selection import cross_val_score
from Class_regression import logreg
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from Class_tree import DecisionTree
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import GridSearchCV
    
import sys
sys.path.append(r"C:\Dinesh\TAMU\Courses\Spring 2018\Midterm")
    
    
    # In[3]:
from sklearn.model_selection import cross_validate
    # In[5]:
    
    
file_path = r"C:\Dinesh\TAMU\Courses\Spring 2018\Midterm"
df = pd.read_excel(file_path+"\CreditCard_Defaults.xlsx")
    
df.dtypes
    # In[13]:
    
    
attribute_map = {
        'Default':[1,(0,1),[0,0]],
        'Gender':[1,(1, 2),[0,0]],
        'Education':[2,(0,1,2,3,4,5,6),[0,0]], 
        'Marital_Status':[2,(0,1,2,3), [0,0]],
        'card_class':[2,(1,2,3),[0,0]],
        'Age':[0,(20,80),[0,0]],
        'Credit_Limit':[0,(100, 80000),[0,0]],
        'Jun_Status':[0,(-2, 8),[0,0]],
        'May_Status':[0,(-2, 8),[0,0]],
        'Apr_Status':[0,(-2, 8),[0,0]],
        'Mar_Status':[0,(-2, 8),[0,0]],
        'Feb_Status':[0,(-2, 8),[0,0]],
        'Jan_Status':[0,(-2, 8),[0,0]],
        'Jun_Bill':[0,(-12000, 32000),[0,0]],
        'May_Bill':[0,(-12000, 32000),[0,0]],
        'Apr_Bill':[0,(-12000, 32000),[0,0]],
        'Mar_Bill':[0,(-12000, 32000),[0,0]],
        'Feb_Bill':[0,(-12000, 32000),[0,0]],
        'Jan_Bill':[0,(-12000, 32000),[0,0]],
        'Jun_Payment':[0,(0, 60000),[0,0]],
        'May_Payment':[0,(0, 60000),[0,0]],
        'Apr_Payment':[0,(0, 60000),[0,0]],
        'Mar_Payment':[0,(0, 60000),[0,0]],
        'Feb_Payment':[0,(0, 60000),[0,0]],
        'Jan_Payment':[0,(0, 60000),[0,0]],
        'Jun_PayPercent':[0,(0, 1),[0,0]],
        'May_PayPercent':[0,(0, 1),[0,0]],
        'Apr_PayPercent':[0,(0, 1),[0,0]],
        'Mar_PayPercent':[0,(0, 1),[0,0]],
        'Feb_PayPercent':[0,(0, 1),[0,0]],
        'Jan_PayPercent':[0,(0, 1),[0,0]]
    }
    
    
    # In[14]:df.drop(['Customer'], axis=1)
    
    # In[6]:
    
    
np.sum(df['Marital_Status']==0)
df.dtypes
    
    
    # In[15]:
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', display=True)
encoded_df = rie.fit_transform(df)
    
    
varlist = df['Default']
X = encoded_df.drop('Default', axis=1)
y = encoded_df['Default']

lgr = LogisticRegression()

#Selecting the best attributes using RFE - 25 attributes chosen   
rfe = RFE(lgr,25)
rfe = rfe.fit(X,y)


print(rfe.support_)
print(rfe.ranking_)


#Code to pick the chosen 25 attribute names and assign it to an array    
a = rfe.ranking_
b = X.columns.values 
c = np.c_[a,b] 
c = pd.DataFrame(c)
c.columns = ['c1','c2']
cols = c.loc[c['c1']== 1,'c2']

#Subset of original dataframe - after RFE
X = X[cols]

# Instantiate the GridSearchCV object: logreg_cv
#logreg_cv = GridSearchCV(lgr, param_grid, scoring = 'roc_auc', cv=5)
#logreg_cv.fit(X, y)

# Print the optimal parameters and best score
#Fitting a LgR Classifier with the best C and Regulzarisation parameter using the training data

c_space = np.logspace(-2, 8, 15)
param_grid = ['l1', 'l2']
score_list = ['accuracy', 'recall', 'precision', 'f1']
#max_f1 = 0
for e in c_space:
    for f in param_grid:
        print("C: ", e, "Regularization method: ", f)
        lgr_CV = LogisticRegression(C=e,penalty=f,random_state=12345)
        lgr_CV= lgr_CV.fit(X, y)
        scores = cross_validate(lgr_CV, X, y, scoring=score_list,cv=10)
        
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            
#print("\nBest based on F1-Score")
#print("Best Number of Estimators (trees) = ", best_estimator)
#print("Best Maximum Features = ", best_max_features)


#Printing model metrics for test data
print("Splitting the dataset and comparing Training and Validation:")

X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)

lgr_train = LogisticRegression(C = 3727593.720314938,penalty='l2',random_state=12345)
lgr_train = lgr_train.fit(X_train, y_train)
#logreg.display_binary_split_metrics(lgr_train, X_train, y_train, X_validate, y_validate)

predict_train = lgr_train.predict(X_train)
predict_validate = lgr_train.predict(X_validate)

conf_matt = confusion_matrix(y_true=y_train, y_pred=predict_train)
conf_matv = confusion_matrix(y_true=y_validate, y_pred=predict_validate)
print("\n")
print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
acct = accuracy_score(y_train, predict_train)
accv = accuracy_score(y_validate, predict_validate)

        
print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
        
print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', \
                      precision_score(y_train,predict_train), \
                      precision_score(y_validate,predict_validate)))
print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                      recall_score(y_train,predict_train), \
                      recall_score(y_validate,predict_validate)))
print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', \
                      f1_score(y_train,predict_train), \
                      f1_score(y_validate,predict_validate)))


#Enhancements - Adopted Deepthi's code
#Finding the best probability threshold by looping over training data and maximizing
#the avg of sensitivity and specificity
metriclist=[]
bestmetric=0
bestthreshold=0
predict_prob=lgr_train.predict_proba(X_train)
for i in range(0,100):
    threshold=(i/100)
    predict_mine=np.where(predict_prob<threshold,0,1)
    k=confusion_matrix(y_train,predict_mine[:,1])
    true_neg=k[0,0]
    true_pos=k[1,1]
    true_pred=true_pos+true_neg
    fals_neg=k[1,0]
    fals_pos=k[0,1]
    totalpred=true_pred+fals_pos+fals_neg
    accu=true_pred/totalpred
    sens=true_pos/(true_pos+fals_neg)
    spec=true_neg/(true_neg+fals_pos)
    metric=(sens+spec)/2
    if metric>bestmetric:
        bestmetric=metric
        bestthreshold=threshold
    metriclist.append(metric)
metricarray=np.array(metriclist)
thresholdarray=np.linspace(0.0, 0.99, num=100)
#Plot
pt.plot(thresholdarray[0:90],metricarray[0:90])
#Printing results for training data using best probability threshold
predict_mine=np.where(predict_prob<bestthreshold,0,1)
k=confusion_matrix(y_train,predict_mine[:,1])
true_neg=k[0,0]
true_pos=k[1,1]
true_pred=true_pos+true_neg
fals_neg=k[1,0]
fals_pos=k[0,1]
totalpred=true_pred+fals_pos+fals_neg
accu=true_pred/totalpred
sens=true_pos/(true_pos+fals_neg)
spec=true_neg/(true_neg+fals_pos)
print("Confusion Matrix : Training")
print(k)
print("Training Accuracy:  %s" %accu)
print("Training Sensitivity:  %s" %sens)
print("Training Specificity:  %s" %spec)

##Printing results for validation data using best probability threshold
predict_prob_val=lgr_train.predict_proba(X_validate)
predict_mine_val=np.where(predict_prob_val<bestthreshold,0,1)
k=confusion_matrix(y_validate,predict_mine_val[:,1])
true_neg=k[0,0]
true_pos=k[1,1]
true_pred=true_pos+true_neg
fals_neg=k[1,0]
fals_pos=k[0,1]
totalpred=true_pred+fals_pos+fals_neg
accu=true_pred/totalpred
sens=true_pos/(true_pos+fals_neg)
spec=true_neg/(true_neg+fals_pos)
print("Confusion Matrix : Validation")
print(k)
print("Validation Accuracy:  %s" %accu)
print("Validation Sensitivity:  %s" %sens)
print("Validation Specificity:  %s" %spec)
