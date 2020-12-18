import pandas as pd
import matplotlib.pyplot as plt

'''
making dummies
'''
salary = pd.read_csv('https://drive.google.com/uc?export=download&id=1kkAZzL8uRSak8gM-0iqMMAFQJTfnyGuh')

dummy = pd.get_dummies(salary['sex'], prefix='sex', drop_first=True)
#k-1개 varible 만들고 싶으면 drop_first를 True, default는 False
#이 경우 첫 value가 male 이기 떄문에 male만 남는다

'''
sex 사용해 mean, std 살펴보기
'''
varname='sex'
gmean = salary.groupby(varname)['salary'].mean()
gmean

gstd = salary.groupby(varname)['salary'].std()
gstd

plt.bar(range(len(gmean)), gmean)
plt.errorbar(range(len(gmean)), gmean, yerr=gstd, fmt='o', c='r', ecolor='r', capthick=2, capsize=3)
# yerr이 빨간 선 범위

'''
rank 사용해 mean, std 살펴보기 (교수 종류)
'''
varname='rank'
gmean = salary.groupby(varname)['salary'].mean()
gmean

gstd = salary.groupby(varname)['salary'].std()
gstd

plt.bar(range(len(gmean)), gmean)
plt.errorbar(range(len(gmean)), gmean, yerr=gstd, fmt='o', c='r', ecolor='r', capthick=2, capsize=3)
plt.xticks(range(len(gmean)), gmean.index)


'''
one way anova
'''
from scipy.stats import f_oneway

salary['rank'].value_counts()
groups = [x[1].values for x in salary.groupby(['rank'])['salary']]
#각 행이 prof, asstProf, AssocProf에 해당하고, 각각 리스트가 있고, 리스트안에 벨류

f_oneway(*groups) # p< 0.01 / means are significantly different

catvar=['rank','discipline','sex']
for c in catvar:
    dummy = pd.get_dummies(salary[c], prefix=c, drop_first=True)
    salary = pd.concat((salary, dummy), axis=1) # combine row-wise

X = salary.drop(catvar+['salary'], axis=1)
y = salary['salary']

import statsmodels.api as sm

X = sm.add_constant(X)

model=sm.OLS(y,X)
result=model.fit()

result.summary()
# r^2 는 작지만 F test의 pvalue가 엄청 작으므로 괜찮은 모델
# sex coef의 t test pvalue는 크다. sex는 별 도움이 안된다 -> 다음 모델링에서 뺴보는 것이 낫다
# rank_AsstProf coef는 음수고, rank_Prof coef는 양수 -> rank_assoc에 비해 각각 평균적으로 적고, 많다. 
# (수업 때 배운 선 두개 있는 그래프 생각)

'''
king county linear regression
'''
house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')

#numeric 과 waterfron(binary)는 그냥 씀
varlist=['bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'sqft_above', 'sqft_basement',
         'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

X1 = house[varlist]
X1=sm.add_constant(X1)
y1 = house['price']

model1 = sm.OLS(y1, X1)
result1 = model1.fit()
result1.summary()

# residual 시각화
y_pred1 = result1.predict(X1)
err1 = y1-y_pred1

import numpy as np #for drawing line
xx = np.linspace(y1.min(), y1.max(), 100)

plt.scatter(y1, y_pred1)
plt.plot(xx,xx,color='k') #y=x 직선
plt.ylabel('Predicted')
plt.xlabel('Real')
# y=x 직선에 비해 점들의 모양의 기울기가 낮다 -> predict < real

plt.hist(err1, bins=50) # 오른쪽으로 long tail. predict << real

from scipy.stats import probplot
probplot(err1, plot=plt) #x축은 std norm dist의 quantile
# 선에 잘 있지만 양 끝이 std norm dist 와 좀 다르다... 
# residual 이 std norm 따른다고 하기 힘들다.

# Breush-pegan test
from statsmodels.stats import diagnostic

diagnostic.het_breuschpagan(err1, X1)
# 앞에 두개는 라그랑스 지수, 뒤에 두개는 F 테스트 값 (값, pvalue, 값, pvalue)
diagnostic.het_breuschpagan(err1, X1[['bedrooms']])
diagnostic.het_breuschpagan(err1, X1[['bedrooms', 'bathrooms']])

'''
outlier 제거하기
싼집과 엄청 비싼 집이 너무 빰!빰!
'''
cond = (house['price']<1000000)&(house['price']>=20000)
X2 = house[cond][varlist]
y2 = house[cond]['price']
X2 = sm.add_constant(X2)



model2 = sm.OLS(y2, X2)
result2 = model2.fit()
result2.summary()

y_pred2 = result2.predict(X2)

# outlier 시각화
y1.plot.kde()
y2.plot.kde() #outlier제거돼서 훨씬 normal 해 보임

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error

mean_squared_error(y1[cond], y_pred1[cond])
mean_squared_error(y2, y_pred2)




