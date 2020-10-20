import numpy as np 
import matplotlib.pyplot as plt 
import datetime 

plt.style.use('ggplot')


ci95intvl = np.array([[203800,245700],[101000,133100],[85600,123400], [46900,75200], [29300,52700], [19300,36700],[20100,37900],[16900,33800], [19000,40700], [18900,40800],[23700,53200],[18500,39900],[15000,34000]])/(7)
median = np.array([224400,116600, 103600, 59800, 39700,27100, 28200, 24600, 28300,28300,35700, 27700, 24000])/(7)
dates = np.array([ [[25,9],[1,10]] ,[[18,9],[24,9]], [[13,9],[19,9]], [[4,9],[10,9]] , [[30,8],[5,9]], [[19,8],[25,8]], [[14,8],[20,8]], [[7,8],[13,8]], [[3,8],[9,8]], [[27,7],[2,8]], [[20,7],[26,7]], [[13,7],[19,7]], [[6,7],[12,7]]])
nicedates = []
for d in dates:
    day1 = [d[0][0] if len(str(d[0][0])) == 2 else '0{}'.format(d[0][0]) ][0]
    day2 = [d[1][0] if len(str(d[1][0])) == 2 else '0{}'.format(d[1][0]) ][0]
    m1 = [d[0][1] if len(str(d[0][1])) == 2 else '0{}'.format(d[0][1]) ][0]
    m2 = [d[1][1] if len(str(d[1][1])) == 2 else '0{}'.format(d[1][1]) ][0]
    nicedates.append( [np.datetime64('2020-{}-{}'.format(m1,day1)),np.datetime64('2020-{}-{}'.format(m2,day2)) ] )

###correct for period lengths

# for nd in range(len(nicedates)):

# print(nicedates)

#going to use median day with +-3 day error on the x axis
# y data and error self explanatory hopefully, non uniform so maybe a bit weird

midday = []
for n in nicedates:
    midday.append(n[0] + (n[1] - n[0])/2 )

xlow = np.array([n[0] for n in nicedates]) - np.array(midday)
xhigh = np.array([n[1] for n in nicedates])- np.array(midday)
xerrors = np.array([np.abs(xlow), xhigh])

ylow = np.array([y[0] for y in ci95intvl]) - np.array(median)
yhigh = np.array([y[1] for y in ci95intvl]) - np.array(median)
yavg = (np.abs(ylow)+yhigh)/2
yerrors = np.array([np.abs(ylow),yhigh])

# print(xerrors, xerrors.shape, yerrors, yerrors.shape)
plt.figure()
plt.ylabel(r'$\mathrm{Daily \ Cases}$', fontsize=14)
ax = plt.gca()
ax.ticklabel_format(axis='y',style='sci')

plt.errorbar(midday, median, xerr=xerrors, yerr=yerrors, ls='none',color='black',capsize=3)

daycnt = np.array(midday) - midday[-1]
print(np.array(daycnt)[0])

# for i in range(len(median)):
#     print(median[i], np.median(ci95intvl[i]))

from lmfit import Parameters, fit_report, minimize, Minimizer
import lmfit

params = Parameters()
params.add('shift', value=20e3)
params.add('factor', value=2)

def residual(params, x,y,yerr):
    vals = params.valuesdict()
    shift = vals['shift']
    factor = vals['factor']
    out = shift + np.exp(factor * x)
    return (out - y) / yerr

mini = lmfit.Minimizer(residual,params, fcn_args=(np.array(daycnt, dtype=int),median, yavg))
out = mini.minimize()
print(fit_report(out))

xmodeldays = np.arange(midday[-1], np.datetime64('2020-10-10'), dtype='datetime64[D]')
xmodeldaysints = np.array(xmodeldays - xmodeldays[0], dtype = int)
modelshift = out.params['shift']
modelfactor = out.params['factor']
modelout = modelshift+np.exp(modelfactor * xmodeldaysints)
plt.plot(xmodeldays, modelout,color='black')

modelshift = out.params['shift']+out.params['shift'].stderr
modelfactor = out.params['factor']+out.params['factor'].stderr
modelout = modelshift+np.exp(modelfactor * xmodeldaysints)
plt.plot(xmodeldays, modelout,color='blue')

modelshift = out.params['shift']-out.params['shift'].stderr
modelfactor = out.params['factor']-out.params['factor'].stderr
modelout = modelshift+np.exp(modelfactor * xmodeldaysints)
plt.plot(xmodeldays, modelout,color='blue')

plt.show()