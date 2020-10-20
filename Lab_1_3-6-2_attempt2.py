import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
import statsmodels 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 

out = pyreadr.read_r('C:\\Users\\abdul\\Google Drive\\Pre_Job_Training\\Introduction_to_Statistical_Learning\\Data_Sets\\Boston.rda')
df = pd.DataFrame(out['Boston'], columns=out['Boston'].keys())

# results = statsmodels.regression.linear_model.OLS(np.array(df['medv']),np.array(df['lstat'])).fit()
# print(results.summary())

# fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
# ax1.scatter(np.array(df['lstat']), np.array(df['medv']))#results.fittedvalues, results.resid)
# ax2.scatter(np.array(df['lstat']), results.resid)
# ax1.plot(np.array(df['lstat']), results.predict(np.array(df['lstat'])))
# plt.show()
x0 = np.array(df['lstat'])
x = sm.add_constant(x0)
y = np.array(df['medv'])

results = sm.OLS(y,x).fit()
print(results.summary())
print(results.conf_int(alpha=0.025))
predicted = results.predict(x)

plt.figure()
plt.scatter(x0,y)
plt.plot(x0,predicted)
plt.show()

from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std

prstd, iv_l, iv_u = wls_prediction_std(results)
st, data, ss2 = summary_table(results, alpha=0.05)

fittedvalues = data[:, 2]
predict_mean_se  = data[:, 3]
predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
predict_ci_low, predict_ci_upp = data[:, 6:8].T

# Check we got the right things
print(np.max(np.abs(results.fittedvalues - fittedvalues)))
print(np.max(np.abs(iv_l - predict_ci_low)))
print(np.max(np.abs(iv_u - predict_ci_upp)))

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(x0, y, 'o')
ax1.plot(x0, fittedvalues, '-', lw=2)
ax1.plot(x0, predict_ci_low, 'r--', lw=2)
ax1.plot(x0, predict_ci_upp, 'r--', lw=2)
ax1.plot(x0, predict_mean_ci_low, 'r--', lw=2)
ax1.plot(x0, predict_mean_ci_upp, 'r--', lw=2)

ax2.scatter(x0, results.resid)
plt.show()

