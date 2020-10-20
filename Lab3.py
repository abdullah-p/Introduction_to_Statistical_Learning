import numpy as np
import pandas as pd 
import statsmodels.api as sm
import scipy 
import scikits.bootstrap as bootstrap

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt 

plt.style.use('ggplot')
np.set_printoptions(precision=4)

### importing data ###
auto = pd.read_csv('C:\\Users\\abdul\\Google Drive\\Pre_Job_Training\\Introduction_to_Statistical_Learning\\Data_Sets\\Auto.csv')
auto = auto.dropna()
print('auto length', len(auto))



###prefilter
auto = auto[auto['horsepower'] != '?']

### seed random number generator
np.random.seed(0)
# 196 observations as training data (this is half moron)
boolarr = np.random.choice([True,False],len(auto))

### generate appropriate data sets
xall = np.array(auto['horsepower'], dtype = float)
xtrain = auto['horsepower'][boolarr]
xtest = auto['horsepower'][~boolarr]
yall= np.array(auto['mpg'])
ytrain = auto['mpg'][boolarr]
ytest = auto['mpg'][~boolarr]

xtrain0 = sm.add_constant(np.array(xtrain,dtype=float))
xall0 = sm.add_constant(xall)

### fit the linear model
results = sm.OLS(np.array(ytrain),xtrain0).fit()
predictout = results.predict(xall0)
mse = np.mean((yall[~boolarr]-predictout[~boolarr])**2)
print('mean square error is', mse)

### fit the quadratic model
from sklearn.preprocessing import PolynomialFeatures
polynomial_features = PolynomialFeatures(degree=2)
x2 = polynomial_features.fit_transform(np.array(xtrain).reshape(-1,1))
deg2 = sm.OLS(np.array(ytrain), x2).fit()
xall0_2 = polynomial_features.fit_transform(xall.reshape(-1,1))
val2 = deg2.predict(xall0_2)
mse2 = np.mean((yall[~boolarr]-val2[~boolarr])**2)
print('polynomial degree 2 mean square error is', mse2)

### fit the order 3 model
from sklearn.preprocessing import PolynomialFeatures
polynomial_features3 = PolynomialFeatures(degree=3)
x3 = polynomial_features3.fit_transform(np.array(xtrain).reshape(-1,1))
print(x3[0:20])
deg3 = sm.OLS(np.array(ytrain), x3).fit()
xall0_3 = polynomial_features3.fit_transform(xall.reshape(-1,1))
val3 = deg3.predict(xall0_3)
mse3 =  np.mean((yall[~boolarr]-val3[~boolarr])**2)
print('polynomial degree 3 mean square error is', mse3)
print(deg3.summary())


# ### testing other code

# x_train = sm.add_constant(np.column_stack((xall[boolarr], xall[boolarr]**2, 
#                                            xall[boolarr]**3 )))
# x_test = sm.add_constant(np.column_stack((xall[~boolarr], xall[~boolarr]**2,
#                                           xall[~boolarr]**3)))

# # make the model and fit
# cube_model = sm.OLS(ytrain, x_train)
# cube_model_results = cube_model.fit()
# print(cube_model_results.summary())
# # use the model on the validation set

# y_predictions = cube_model_results.predict(x_test)

# print('\nCubic Model MSE = ', np.mean((ytest.values-y_predictions)**2))


### leave one out cross validation ###

# np.newaxis basically does that thing we like where every value become a mini sub array
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

x_loocv = xall[:,np.newaxis]
# use yall

orders = np.arange(1,6)
mse_est = []

for index, order in enumerate(orders):
    polytemp = PolynomialFeatures(degree=order, include_bias=False)
    xtofit = polytemp.fit_transform(x_loocv)
    # cant use statsmodel anymore :( 
    out = LinearRegression()
    out.fit(xtofit,yall)
    mse_est.append( np.mean( -1 * cross_val_score(out, xtofit, yall, scoring='neg_mean_squared_error', cv=len(xtofit)) ) )
    #loss so multiply by -1
print('\nThe estimated test MSEs = ', mse_est)

fig, ax = plt.subplots()
ax.plot(orders, mse_est)
ax.set_xlabel('Polynomial Order')
ax.set_ylabel('LOOCV Error Approx')
plt.show()