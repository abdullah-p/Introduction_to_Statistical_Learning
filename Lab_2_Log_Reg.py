### Chapter 4, Logistic Regression Lab ###

import numpy as np 
import pandas as pd 
import pyreadr
import statsmodels.api as sm
import statsmodels 
import statsmodels.formula.api as smf

smarket =pd.read_csv('C:\\Users\\abdul\\Google Drive\\Pre_Job_Training\\Introduction_to_Statistical_Learning\\dataset-18213.csv')
print(smarket.head())
print(smarket.corr()) ## only volume and year have significant correlations

### create dummy variable 
smarket['Up'] = np.where(smarket['Direction']=='Up',1,0)

### isolate predictors and response 
x = smarket[['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']].values ## couldve just kept these as dataframes
y = smarket['Up'].values

train_bool = smarket['Year'].values < 2005
x_train = x[train_bool]
x_test = x[~train_bool]
y_train = y[train_bool]
y_test = y[~train_bool]

results = smf.logit('Up ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', data=smarket[smarket['Year'].values<2005]).fit()
print(results.summary())
print(results.pred_table())
confusionout = np.array(results.pred_table())
print('training error rate', 1-np.trace(confusionout)/np.sum(confusionout))


predict = results.predict(pd.DataFrame(x_test, columns=['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']))
predict = np.where(predict>0.5,1,0)

from sklearn.metrics import confusion_matrix
print(y_test)
conf = confusion_matrix(y_test, predict)
print(conf)
print('test error rate', 1 - np.trace(conf)/np.sum(conf))

### trying it with just the 2 previous days
results_2 = smf.logit('Up ~ Lag1 + Lag2', data=smarket[smarket['Year'].values<2005]).fit()
print(results_2.summary())
print(results_2.pred_table())
confusionout2 = np.array(results_2.pred_table())
print('training error rate', 1-np.trace(confusionout2)/np.sum(confusionout2))

predict2 = results_2.predict(pd.DataFrame(x_test, columns=['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']))
predict2 = np.where(predict2>0.5,1,0)

print(y_test)
conf2 = confusion_matrix(y_test, predict2)
print(conf2)
print('test error rate', 1 - np.trace(conf2)/np.sum(conf2))

predict3 = results_2.predict( pd.DataFrame([[1.2,1.1],[1.5,-0.8]],columns=['Lag1','Lag2']))
print(predict3) # fucking hell i've done it right

### linear discriminant analysis ###

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

print('Linear Discriminant Analysis')
lda = LinearDiscriminantAnalysis()
lda.fit(x_train[:,:2],y_train)
print(confusion_matrix(y_test,lda.predict(x_test[:,:2])))

print('Quadratic Discriminant Analysis')
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train[:,:2],y_train)
print(confusion_matrix(y_test,qda.predict(x_test[:,:2])))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train[:,:2],y_train)
print(confusion_matrix(y_test, knn.predict(x_test[:,:2])))
# transpose of the R way 