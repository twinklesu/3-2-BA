import pandas as pd

data = pd.read_csv('https://drive.google.com/uc?export=download&id=1gd2jStJinE_egX7LCKnh1-_lxafRI-zF')

data.columns
data['default.payment.next.month'].value_counts() #1의 크기가 훨씬 작다
data['default.payment.next.month'].value_counts()/len(data) #0.22밖에 안됨

X = data[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
y = data['default.payment.next.month']

catvar = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for c in catvar:
    dummy = pd.get_dummies(X[c], prefix=c, drop_first=True)
    X = pd.concat((X,dummy), axis=1)
X = X.drop(catvar, axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_curve

clf = LogisticRegression(max_iter = 300)

trnX, valX, trnY, valY = train_test_split(X,y, test_size=0.2, random_state=10)

clf.fit(trnX, trnY)

clf.score(trnX, trnY)
clf.score(valX, valY) # not bad, but

from sklearn.metrics import confusion_matrix

y_pred = clf.predict(trnX)
confusion_matrix(trnY, y_pred) # no sample class 1 correctly classified as 1

y_pred = clf.predict(valX)
confusion_matrix(valY, y_pred)

recall_score(valY, y_pred) #0이다
precision_score(valY, y_pred)

y_prob = clf.predict_proba(valX)

import matplotlib.pyplot as plt
plt.boxplot([y_prob[valY==0,1], y_prob[valY==1,1]])

'''threshold 바꿈'''
y_pred2 = (y_pred[:,1]>0.3)*1

confusion_matrix(valY, y_pred2)

recall_score(valY, y_pred2) #better
precision_score(valY, y_pred2)
f1_score(valY, y_pred2)

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection, TomekLinks

ros = RandomOverSampler(random_state = 0, sampling_strategy = 'auto')
rus = RandomUnderSampler(random_state = 0, sampling_strategy = 'auto')
sm = SMOTE(random_state=0, k_neighbors=5)
ada = ADASYN(random_state=0, n_neighbors=5)
nm1 = NearMiss(version=1)
nm2 = NearMiss(version=2)
nm3 = NearMiss(version=3)
oss = OneSidedSelection(random_state=0, n_neighbors=1, n_seeds_S=1)
tl = TomekLinks(sampling_strategy='all')

result = pd.DataFrame(columns=['Sampling', 'Acc', 'Recall', 'Precision', 'F1'])

y_pred = clf.predict(valX)
result.loc[0] = ['Original']+[x(valY,y_pred) for x in [accuracy_score, recall_score, precision_score, f1_score]]

sampler = {'ros':ros, 'rus':rus, 'sm':sm, 'ada':ada, 'nm1': nm1, 'nm2':nm2, 'nm3':nm3, 'tl':tl, 'oss':oss}

count=1
for name, s in sampler.items():
    X_resampled, y_resampled = s.fit_sample(trnX, trnY)
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(valX)
    result.loc[count] = [name]+[x(valY,y_pred) for x in [accuracy_score, recall_score, precision_score, f1_score]]
    count+=1
 
    
    
clf.fit(trnX, trnY)
y_pred = clf.predict(valX)
y_prob = clf.predict_proba(valX)
fpr, tpr, thres = roc_curve(valY, y_prob[:,1])
roc = [[fpr, tpr]]
for name, s in sampler.items():
    X_resampled, y_resampled = s.fit_sample(trnX, trnY)
    clf.fit(X_resampled, y_resampled)
    y_prob = clf.predict_proba(valX)
    fpr, tpr, thres = roc_curve(valY, y_prob[:,1])    
    roc.append([fpr, tpr])
    
from itertools import cycle
fig=plt.figure(figsize=(10,8))
lines = cycle(['-', '-.', '--', ':'])
ax = plt.gca()
for name, (fpr, tpr), ls in zip(['Origianl']+list(sampler.keys()), roc, lines):
    plt.plot(fpr, tpr, label=name, linestyle=ls)
plt.legend(fontsize=14)

import numpy as np

np.random.seed(0)
n_samples_1 = 1000
n_samples_2 = 100

X2 = np.r_[1.5*np.random.randn(n_samples_1, 2), 0.5*np.random.randn(n_samples_2, 2) +[2,2]]
y=[0]*(n_samples_1) + [1]*(n_samples_2)

plt.scatter(X2[:,0], X2[:,1],c=y)

clf.fit(X2,y)

w=clf.coef_[0]
xx = np.linspace(-5,5,100)
yy = -w[0]/w[1]*xx-clf.intercept_[0]/w[1]

plt.scatter(X2[:,0], X2[:,1],c=y)
plt.plot(xx,yy)
    

wclf = LogisticRegression(max_iter=300, class_weight={1:5})
wclf.fit(X2,y)

ww = wclf.coef_[0]
wyy = -ww[0]/ww[1]*xx-wclf.intercept_[0]/ww[1]
plt.scatter(X2[:,0], X2[:,1],c=y)
plt.plot(xx,wyy) #decision boundary가 내려갔다. 


from sklearn.neighbors import LocalOutlierFactor
np.random.seed(42)

X_inliers = np.random.randn(100,2)
X_inliers = np.r_[0.3*X_inliers+2, 0.8*X_inliers-2]

X_outliers = np.random.uniform(low=-4, high=4, size=(20,2))

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:]=-1

plt.scatter(X[:,0], X[:,1], c=ground_truth) #교수님은 되는데 왜 안되지ㅜ

lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
y_pred = lof.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=ground_truth) # ㅜㅜ
plt.scatter(X[y_pred==-1,0],X[y_pred==-1,1], marker='x', c='r')
