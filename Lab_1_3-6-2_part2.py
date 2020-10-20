import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
import statsmodels 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 

out = pyreadr.read_r('C:\\Users\\abdul\\Google Drive\\Pre_Job_Training\\Introduction_to_Statistical_Learning\\Data_Sets\\Boston.rda')
df = pd.DataFrame(out['Boston'], columns=out['Boston'].keys())

# out = []
# for i in range(len(y[0])):
#     temp = []
#     for j in range(len(y)):

# string_cols = ' + '.join(df.columns[:-1])
# results = smf.ols('medv ~ {}'.format(string_cols), data = df).fit()
y = df['medv']
x0 = df[df.columns[:-1]]
x = sm.add_constant(x0)
print(x0.shape, y.shape)

results = sm.OLS(y,x).fit()
print(results.summary())

### now drop age ###
x0_1 = df[  df.columns[:-1][~(df.columns[:-1] == 'age')]  ]
x_1 = sm.add_constant(x0_1)
results2 = sm.OLS(y,x_1).fit()
print(results2.summary())

### interaction ### - will have to use statsmodel formula api
results3 = smf.ols('medv ~ lstat * age', data=df).fit()
print(results3.summary())

### non linear transformation of predictor 

from statsmodels.stats.anova import anova_lm
results5 = smf.ols('medv ~ lstat + np.power(lstat,2)', data=df).fit()
results4 = smf.ols('medv ~ lstat ', data=df).fit()

print(results4.summary(), results5.summary())
 ### use anova to quantify how much better quadratic fit is to linear fit
 ### conducts a hypothesis test
 ### null is same fit to data, alternative is that quadratic is better

print(anova_lm(results4, results5))
 ### near 0 p value and f statistic of 135 provide good evidence of superior fit

### polynomial

from sklearn.preprocessing import PolynomialFeatures
polynomial_features = PolynomialFeatures(degree=5)
x0_2_toplt =  np.array(df['lstat'])
x0_2 = np.array(df['lstat']).reshape(-1,1) ## we have single feature and lots of samples
print(x0_2)
# x = sm.add_constant(x0)
xp = polynomial_features.fit_transform(x0_2)

results6 = sm.OLS(y,xp).fit()
print(results6.summary())
ypred = results6.predict(xp)
plt.figure()
plt.scatter(x0_2_toplt, y)
args = np.argsort(x0_2_toplt) # avoid sorting by just creating dummy x range from higest to lowest
x0_2_toplt = x0_2_toplt[args]
ypred = ypred[args]
plt.plot(x0_2_toplt, ypred)
plt.show()
