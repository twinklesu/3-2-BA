from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = datasets.load_diabetes()
X = data.data
y = data.target

X /= X.std(axis=0) #normalization

trainX, validX, trainY, validY = train_test_split(X,y, test_size=0.2, shuffle=True, random_state=30)

reg1 = LinearRegression()
reg2 = Ridge(alpha=1) #alpha는 공식에서 감마. panelty term
reg3 = Lasso(alpha=1)

reg1.fit(trainX, trainY)
reg1.coef_
reg2.fit(trainX, trainY)
reg2.coef_ #절댓값이 감소한 값들을 확인할 수 있다
reg3.fit(trainX, trainY)
reg3.coef_ #sparse solution. 0이 많다

'''panelty term 확인'''
alphas = np.logspace(-3,3,30)

linear_r2 = reg1.score(validX, validY) #constant. alpha와 무관
result = pd.DataFrame(index=alphas, columns=['Ridge','Lasso'])
for alpha in alphas:
    reg2.alpha = alpha
    reg3.alpha = alpha
    reg2.fit(trainX, trainY)
    result.loc[alpha, 'Ridge'] = reg2.score(validX, validY)
    reg3.fit(trainX, trainY)
    result.loc[alpha, 'Lasso'] = reg3.score(validX, validY)

# 그래프
plt.plot(np.log(alphas), result['Ridge'], label='Ridge')
plt.plot(np.log(alphas), result['Lasso'], label='Lasso')
plt.hlines(linear_r2, np.log(alphas[0]), np.log(alphas[-1]), ls=":", colors='k', label='Ordinary')
plt.legend() #어느 시점에 잠깐 Ridge 와 Lasso가 더 좋은 성능을 보인다

##
X, y = datasets.make_regression(n_samples=1000, n_features = 10, n_informative=10)
X.mean(axis=0)
X.std(axis=0) #std norm dist

reg1.fit(X,y)
reg1.coef_
reg2.alpha=10 #alpha가 커져서 beta 값이 작아짐
reg2.fit(X,y)
reg2.coef_

beta1 = reg2.coef_

X2 = X.copy()
X2[:,0]/=10

reg1.fit(X2,y) #rescaling 한 곳이 10배 됐음
reg1.coef_

reg2.fit(X2,y)
beta2 = reg2.coef_

beta1
beta2

beta1/beta2 #다른 숫자들은 1에 가가운데 첫 feature만 beta2가 매우 큰것으로 보임

'''Elastic Net Regression'''
from sklearn.linear_model import lasso_path, enet_path
data = datasets.load_diabetes()
X = data.data
y = data.target
X /= X.std(axis=0) #normalization

eps = 5e-3
alphas_lasso, coefs_lasso, _ = lasso_path(X,y,eps,fit_intercept=False)
alphas_enet, coefs_enet, _ = enet_path(X,y,eps=eps,l1_ratio=0.5, fit_intercept = False)
# l1 ratio가 커질수록 lasso의 비중이 커짐.
from itertools import cycle

colors = cycle(plt.cm.tab10(np.arange(10)/10))

neg_log_alphas_lasso = -np.log10(alphas_lasso) 
neg_log_alphas_enet = -np.log10(alphas_enet)

for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, c=c, ls='--')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'ElasticNet'))
plt.axis('tight')
# elastic이 less sparse

#%% Rossmann
import pandas as pd

train = pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', dtype={'StateHoliday':'str'})
store = pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')

train = train.merge(store, on=['Store'])

sales = train[['Store', 'Date', 'Sales', 'Open']]
sales_count = sales.groupby('Store')['Date'].count() #개점 날짜가 다르다
sales = sales[sales['Store'].isin(sales_count[sales_count==sales_count.max()].index)]
sales['Date'] = pd.to_datetime(sales['Date']) #원래 그냥 string 이였는데 시간으로 바뀜

daily_sales = sales[sales['Open']==1].groupby(['Date'])['Sales'].mean()
plt.plot(daily_sales) #완만하게 만들기 위해 moving avg 사용

# 여기서 부터 잠깐 다른걸로 예시
s = pd.Series(np.random.randn(1000), index = pd.date_range('1/1/2000', periods = 1000))
s = s.cumsum()
plt.plot(s)
r = s.rolling(window=60) #6개의 점으로 mean 계산
s.plot(style='k--')
r.mean().plot(style='r') #smoother

# moving avg 이용
fig = plt.figure(figsize=(10,8))
plt.plot(daily_sales, ':k')
plt.plot(daily_sales.rolling(window=30).mean(), 'r')

train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month

sel_store = sales_count[sales_count==sales_count.max()].index
sel_train = train[train['Store'].isin(sel_store)]
sel_train = sel_train.sort_values(['Store', 'Date'])

catvar = ['DayOfWeek', 'Month', 'StoreType', 'Assortment', 'StateHoliday']
for c in catvar:
    temp = pd.get_dummies(sel_train[c], prefix=c, drop_first=True)
    sel_train = pd.concat((sel_train, temp), axis=1)
    
sel_train = sel_train.drop(catvar, axis = 1)
sel_train = sel_train[sel_train['Open']==1]

trainX = sel_train[sel_train['Date'] <= pd.to_datetime('20141231')]
valX = sel_train[sel_train['Date']>pd.to_datetime('20141231')] #2015년 데이터

remove_cols = ['Store', 'Date', 'Open', 'Sales', 'Year', 'CompetitionDistance', 
               'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
               'Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval', 'Customers']

trainY = trainX['Sales']
trainX = trainX.drop(remove_cols, axis=1)

valY = valX['Sales']
valX = valX.drop(remove_cols, axis=1)

reg1 = LinearRegression()
reg1.fit(trainX, trainY)
reg1.score(trainX, trainY)
reg1.score(valX, valY)

sel_store_history = sel_train[sel_train['Store'].isin(np.random.choice(sel_store,10))]
sel_store_history = pd.pivot_table(sel_store_history, index='Date', columns='Store', values='Sales')
sel_store_history = sel_store_history.fillna(0)

for c in sel_store_history.columns:
    plt.plot(sel_store_history[c].rolling(window=30).mean())

# 새로운 방식
sel_train = sel_train.set_index('Date')

# 최근 과거 며칠간의 평균값을 이용
new_variables = sel_train.groupby('Store')['Sales'].rolling(window='7D').mean()

new_variables = new_variables.to_frame().rename(columns={'Sales':'Sales1W'})
new_variables['Sales2W'] = sel_train.groupby('Store')['Sales'].rolling(window='14D').mean()
new_variables['Sales1_2_diff'] = new_variables['Sales1W']-new_variables['Sales2W']
new_variables['Sales1_2_ratio'] = new_variables['Sales1W']/new_variables['Sales2W']

new_variables.head()

new_variables = new_variables.reset_index()
new_variables['Date'] = new_variables['Date'] + pd.to_timedelta('7D') #y에 맞춰 미룬듯?

new_sel_train = sel_train.merge(new_variables, on=['Store', 'Date'], how='left')

#초반 2주 데이터는 지운다
new_sel_train = new_sel_train[new_sel_train['Date']>=pd.to_datetime('2013-01-15')]

trainX2 = new_sel_train[new_sel_train['Date'] <= pd.to_datetime('20141231')]
valX2 = new_sel_train[new_sel_train['Date']>pd.to_datetime('20141231')] #2015년 데이터
remove_cols = ['Store', 'Date', 'Open', 'Sales', 'Year', 'CompetitionDistance', 
               'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
               'Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval', 'Customers']
trainY2 = trainX2['Sales']
trainX2 = trainX2.drop(remove_cols, axis=1)
valY2 = valX2['Sales']
valX2 = valX2.drop(remove_cols, axis=1)

reg2 = LinearRegression()
reg2.fit(trainX2, trainY2) #왜????NaN왜???
reg2.score(trainX2, trainY2)
reg2.score(valX2, valY2)


# historical customer,,, lasso, ridge,,,, featrue 제거,,, 등등