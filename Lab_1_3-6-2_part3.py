import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
import statsmodels 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 

carseats = pyreadr.read_r('C:\\Users\\abdul\\Google Drive\\Pre_Job_Training\\Introduction_to_Statistical_Learning\\ISLR_1.2.tar\\data\\Carseats.rda')
car = pd.DataFrame(carseats['Carseats'], columns=carseats['Carseats'].keys())
cols = car.columns
all_cols_str = ''
for i in range(len(cols)):
    if str(cols[i]) == 'Sales': ### need to get rid of sales, o.e. regresses perfectly onto self ###
        pass
    else:
        all_cols_str += (str(cols[i]) + ' + ')
print(all_cols_str)
results = smf.ols('Sales ~ {} + Income:Advertising + Price:Age '.format(all_cols_str), data=car).fit()
print(results.summary())