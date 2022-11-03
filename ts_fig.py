
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.ticker as plticker
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,month_plot,quarter_plot
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('~/Desktop/ts_raw.csv')[:600]
#df['Dates'] = pd.to_datetime(df['Dates'])
sns.set_style("whitegrid")


#plt.rcParams["figure.figsize"] = (15,3)
# plot it
plt.clf()
f, (a0, a1,a2,a3) = plt.subplots(4, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1,3,1]})

a2.plot(df['Dates'].values, df['1'], linestyle='dashed',label='Actual' )
a2.plot(df['Dates'].values, df['1.1'], label='SBT'  )

a3.plot(df['Dates'].values, df['4'], linestyle='dashed', label='Actual' )
a3.plot(df['Dates'].values, df['4.1'], label='SBT' )

a0.plot(df['Dates'].values, df['1'], linestyle='dashed', label='Actual' )
a0.plot(df['Dates'].values, df['1.3'],color='green', label='Pyraformer' )

a1.plot(df['Dates'].values, df['4'], linestyle='dashed', label='Actual' )
a1.plot(df['Dates'].values, df['4.3'],color='green', label='Pyraformer' )


a0.set_xlim(left=0)
a1.set_xlim(left=0)
a2.set_xlim(left=0)
a3.set_xlim(left=0)
a0.set_xlim(right=600)
a1.set_xlim(right=600)
a2.set_xlim(right=600)
a3.set_xlim(right=600)
#for tick in a3.xaxis.get_ticklabels():
    #print(tick.get_text()[:8])
    #tick.set_text("asdf")

#a3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

loc = plticker.MultipleLocator(96*2) # this locator puts ticks at regular intervals
a0.xaxis.set_major_locator(loc)
a1.xaxis.set_major_locator(loc)
a2.xaxis.set_major_locator(loc)
a3.xaxis.set_major_locator(loc)

def func(x,pos):
    if pos==1:
        return "10/24/17"
    elif pos==2:
        return '10/26/17'
    elif pos==3:
        return '10/28/17'
    elif pos==4:
        return '10/30/17'
#so annoying
a3.xaxis.set_major_formatter(plticker.FuncFormatter(func))



a0.xaxis.set_ticklabels([])
a1.xaxis.set_ticklabels([])
a2.xaxis.set_ticklabels([])

a0.tick_params(axis='both', which='major', labelsize=16)
a1.tick_params(axis='both', which='major', labelsize=16)
a2.tick_params(axis='both', which='major', labelsize=16)
a3.tick_params(axis='both', which='major', labelsize=16)

i=0
for tick in list(a3.get_xticklabels()):
    tick.set_fontsize(18)
    #labels.append(tick.get_text()[:8], )
    print(tick)
    if i==1:
        tick.set_horizontalalignment("left")
    i+=1




f.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Time", fontsize=20)
#plt.ylabel("Value", fontsize=20)
a0.set_ylabel('HULL', fontsize=18)
a2.set_ylabel('HULL', fontsize=18)

a1.set_ylabel('LUFL', fontsize=18)
a3.set_ylabel('LUFL', fontsize=18)

lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lines=[lines[0],lines[1],lines[-1]]
labels=[labels[0],labels[1],labels[-1]]
f.legend(lines, labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.42, -.02, 0.0, 0), prop={'size': 24})

a0.spines['top'].set_visible(True)
a1.spines['top'].set_visible(False)
a2.spines['top'].set_visible(True)
a3.spines['top'].set_visible(False)
a3.spines['right'].set_visible(True)
a3.spines['bottom'].set_visible(True)
a2.spines['bottom'].set_visible(False)
a3.spines['left'].set_visible(True)

f.savefig('ts.pdf', bbox_inches='tight')