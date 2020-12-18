from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# created imbalanced sample
X,y = make_classification(n_samples=5000, n_features=2, n_informative=2, 
                          n_redundant=0, n_repeated=0, n_classes=3, 
                          n_clusters_per_class =1, weights=[0.01, 0.05, 0.94],
                          class_sep=0.8, random_state=0)

plt.scatter(X[:,0], X[:,1], c=y)


''''simple over sampling'''
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy='auto',random_state=0) # 교수님이랑 버전이 달라서 요로케 해야됨,,


X_resampled, y_resampled = ros.fit_resample(X,y) # 새버전 ind 없음

np.bincount(y) # number of sample in each cluster 
np.bincount(y_resampled) #모든 cluster가 갯수가 같다
# X 갯수 늘어난것도 확인 가능

# 그래프 그리기,,
'''
num_samples = np.bincount(ind)
plt.scatter(X[:,0], X[:,1], c=y, s=num_samples)
'''

'''simple under sampling'''
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0, sampling_strategy='auto')
X_resampled, y_resampled = rus.fit_resample(X,y)

np.bincount(y_resampled) #가장 작았던 수에 맞춰짐

#그래프
plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)



rus = RandomUnderSampler(random_state=0, sampling_strategy={1:64*2,2:64*10})
X_resampled, y_resampled = rus.fit_resample(X,y)

np.bincount(y_resampled)


'''SMOTE'''
from imblearn.over_sampling import SMOTE

sm = SMOTE(k_neighbors=5, random_state=0)
X_resampled, y_resampled = sm.fit_resample(X,y)

np.bincount(y_resampled)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)


'''ADASYN'''
from imblearn.over_sampling import ADASYN

ada = ADASYN(random_state = 0, n_neighbors=5)
X_resampled, y_resampled = ada.fit_resample(X,y)

np.bincount(y_resampled)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)
#SMOTE와 약간 다르게 생김. boundary가 더 깔끔한 편

'''NearMiss'''
from imblearn.under_sampling import NearMiss
nm = NearMiss(version=1)
X_resampled, y_resampled = nm.fit_resample(X,y)

np.bincount(y_resampled)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)

'''one sided'''
from imblearn.under_sampling import OneSidedSelection

oss = OneSidedSelection(random_state = 0, n_neighbors=1, n_seeds_S=1)

X_resampled, y_resampled = oss.fit_resample(X,y)

np.bincount(y_resampled)

'''TomekLink'''
from imblearn.under_sampling import TomekLinks
t1 = TomekLinks(sampling_strategy='all')
#ind가 필요한데 난 ind가 없는걸?ㅜㅜ

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X,y)

xmin, xmax, ymin, ymax = X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()
xx, yy = np.meshgrid(np.linspace(xmin-0.5, xmax+0.5, 10), np.linspace(ymin-0.5, ymax+0.5,10))
zz = np.c_[xx.ravel(), yy.ravel()]
zz_pred = clf.predict(zz)

plt.contourf(xx,yy,zz_pred.reshape(xx.shape), alpha=0.7)
plt.scatter(X[:,0],X[:,1],c=y)




