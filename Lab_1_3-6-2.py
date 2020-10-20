import numpy as np
import pandas as pd
import pyreadr
from lmfit import minimize, Parameters, fit_report
import lmfit
# from sklearn.linear_model import LinearRegression
out = pyreadr.read_r('C:\\Users\\abdul\\Google Drive\\Pre_Job_Training\\Introduction_to_Statistical_Learning\\Data_Sets\\Boston.rda')
df = pd.DataFrame(out['Boston'], columns=out['Boston'].keys())

x = np.array(df['lstat'])
y = np.array(df['medv'])
# X = [[val] for val in np.array(df['lstat'])]
# print(X)
# fit = LinearRegression().fit(X,np.array(df['medv'])) # x lstat, y medv
# print('Intercept is', fit.intercept_)

def residual(params, x, y):
    m = params['gradient']
    c = params['intercept']
    model = m * x + c 
    return (y-model)

def rsqr(out, y):
    ''' no built in r squared functionality to lmfit '''
    return 1 - out.redchi/np.var(y, ddof=2)

# def tstat():
#     # standard error is sigma/sqrt(n)
#     se = numpy.std(y)/numpy.sqrt(len(y))
#     return out.params['gradient'] / 

params = Parameters()
params.add('gradient', value=1)
params.add('intercept', value=1)

mini = lmfit.Minimizer(residual,params, args=(x,y))
out = mini.minimize()#residual, params, args=(x,y))
print(fit_report(out))
print('Residual Standard Error = reduced chi-sqr sqrt', np.sqrt(out.redchi)) # RSE, redchi has N_values - N_DOF in denominator
print('R-squared', rsqr(out,y))

modm= out.params['gradient']
modc = out.params['intercept']
moddata = modm*x + modc

print('testing gradient standard error',out.params['gradient'].stderr)
print('manually this is', np.sqrt(np.std(y)**2 / np.sum( (x-np.mean(x))**2 ) ) )
import matplotlib.pyplot as plt 

# plt.figure()
# plt.scatter(x,y)
# plt.plot(x,moddata)
# plt.show()

ci, trace = lmfit.conf_interval(out, sigmas=[1, 2], trace=True)
lmfit.printfuncs.report_ci(ci)

