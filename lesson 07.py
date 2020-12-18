from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn import datasets
import numpy as np

'''regression'''
boston = datasets.load_boston()
Xb = boston.data
yb = boston.target

f_result = f_regression(Xb, yb) #0은 F, 1은 p-value

order = np.argsort(f_result[0]) #return index of sorted array 뒤에가 큰거

mutual_info_regression(Xb, yb, discrete_features=False)

'''classification'''
iris = datasets.load_iris()
Xi = iris.data
yi = iris.target

f_classif(Xi, yi) # 첫번째는 F 값 두번째는 p-value

mutual_info_classif(Xi, yi, discrete_features=False, n_neighbors=3)

# chi2 하려면 categorize 해줘야 한다.
Xcat = np.zeros(Xi.shape, dtype=int)
for i in range(Xcat.shape[1]):
    h = np.histogram(Xi[:,i], bins=5) # 첫번째 줄은 group별 갯수, 두번째는 bin의 boundary
    Xcat[:,i] = np.sum(Xi[:,[i]]>h[1][1:], axis=1)
    
chi2(Xcat, yi)

from sklearn.feature_selection import SelectKBest
fs = SelectKBest(chi2, k=2)
fs.fit(Xcat, yi)

Xred = fs.transform(Xcat) #reduce X to the selected features

'''election data model'''
import pandas as pd

elec=pd.read_csv('https://drive.google.com/uc?export=download&id=1fq9qDqXLiUm0un_saxAUpPsSJa05F_bV', index_col=0)
county=pd.read_csv('https://drive.google.com/uc?export=download&id=1LciKFXkb3MmpXFEHDk1Db8YFsK0liF3a')

data = elec[elec['county_name']!='Alaska'].merge(county, left_on='FIPS', right_on='fips', how='left')
data_ak = elec[elec['county_name']=='Alaska'].drop_duplicates(['votes_dem_2016', 'votes_gop_2016'])
data_ak['FIPS']=2000
data_ak = data_ak.merge(county, left_on = 'FIPS', right_on='fips', how='left')
data = pd.concat((data, data_ak), axis=0).sort_values('fips')
data['target'] = (data['votes_dem_2016']>data['votes_gop_2016'])*1
data['target'].value_counts()

cols = county.columns[3:].values

Fscore = f_classif(data[cols], data['target'])
MI = mutual_info_classif(data[cols], data['target'])

Fscore = pd.DataFrame({'F': Fscore[0], 'pvalue': Fscore[1]}, index=cols)
Fscore = Fscore.sort_values('F', ascending=False)

MI = pd.DataFrame({'MI': MI}, index=cols)
MI = MI.sort_values('MI', ascending=False)

ks = [10, 20, 30, 40, 50]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
cv=5
clf = LogisticRegression(C=1)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

metr = [accuracy_score, recall_score, precision_score, f1_score]

result = pd.DataFrame(columns=['N_features', 'Fold', 'Acc', 'Recall','Precision', 'F1'])

for s in ks:
    selvars = Fscore.index[:s]
    for pos, (train, valid) in enumerate(skfold.split(data[selvars], data['target'])):
        clf.fit(data.iloc[train][selvars], data.iloc[train]['target'])
        y_pred = clf.predict(data.iloc[valid][selvars])
        result.loc[len(result)] = [s, pos+1]+[m(data.iloc[valid]['target'], y_pred) for m in metr]

result.groupby(['N_features'])['Acc', 'Recall', 'Precision', 'F1'].mean()

Fscore['Group'] = [x[:3] for x in Fscore.index]

group_f = Fscore.groupby(['Group'])['F'].mean()
group_f = group_f.sort_values(ascending=False)

'''feature selection: wrapper'''
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
clf = LogisticRegression(C=1, max_iter=300)

sfs = SFS(clf, k_features=10, forward=True, floating=False, scoring='f1', cv=5)
sfs.fit(data[cols], data['target']) # 시간 좀 오래걸림...교수님은 맥북이라 빨리돌아가네,,,

sfs.subsets_

