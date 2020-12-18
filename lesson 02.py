import pandas as pd
import matplotlib.pyplot as plt

house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')

#%% numerical
house['bedrooms'].describe()

'''
box plot
'''
fig = plt.figure(figsize=(10,8))
house['bedrooms'].plot.box()

fig = plt.figure(figsize=(10,8))
house['bathrooms'].plot.box()

plt.boxplot(house['bathrooms'], whis=(0,100)) # 0 quantile = min, 100 quantile = max

'''
histogram
'''

house['bedrooms'].plot.hist(bins=20)

'''
Kernel Density Estimati
'''
house['bedrooms'].plot.kde() #너무 집중되어 있으므로 window size 올려야 함
house['bedrooms'].plot.kde(bw_method=2)

'''
묶어서 그래프 그리기
'''
house.groupby('yr_built')['price'].mean().plot() #가로는 지어진 년도, 세로는 금액.
house[house['yr_renovated']>0].groupby('yr_renovated')['price'].mean().plot() 
#새로 지어진 경우, 새로지어진 년도와 가격의 관계

'''
scatter plot
'''
from pandas.plotting import scatter_matrix
scatter_matrix(house[['price', 'bedrooms','bathrooms']], figsize=(10,10))
#diagnol에는 histogram, 외엔 scatter plot. 중복되는 정보를 준다면 하나 제거

plt.scatter(house['bedrooms'], house['price']) #outlier 확인

'''
corr 확인
'''
corr = house[['price', 'bedrooms','bathrooms','sqft_living','sqft_lot', 'floors']].corr()
#sqft_living 가 price(output)과 가장 corr함
cax=plt.imshow(corr, vmin=-1, vmax=1, cmap=plt.cm.RdBu) #cov는 -1~1의 값, cmap은 색
plt.colorbar(cax) #cov를 시각적으로 나타냄


#%% Categorical

freq = house['grade'].value_counts()

'''
bar plot
'''
house['waterfront'].value_counts().plot.bar()

'''
category 별로 box plot
'''
house.boxplot(column=['price'], by='waterfront')


#%% Linear Regression
import statsmodels.api as sm
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target

X = sm.add_constant(X) #모든 row에 1인 새 col이 추가된다

model = sm.OLS(y, X) #ordinary Linear Regeression model
result = model.fit() 

result.summary()
