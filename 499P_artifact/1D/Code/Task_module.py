
# coding: utf-8

# # Exploring Visualization Error Metrics
# - The following notebook compares sample visualization error metrics as a graph for different 1D noisy data outputs
# - Generates and compares L1, L2, and Linf errors as well as self-defined M1, M2, M3 errors
# <br>
# M1: Difference of true vs noisy max INDEX
# <br>
# M2: Difference of true vs noisy max VALUE
# <br>
# M3: Difference of true vs noisy 2nd max INDEX

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import math

import sys,os
DPCOMP_PATH = '/nfs/avid/data1/miklau/dpcomp-parent/dpcomp_core_op'
sys.path.append(DPCOMP_PATH)
os.environ['DPCOMP_CORE']= DPCOMP_PATH


# In[618]:

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
sns.set_style("whitegrid")


# In[619]:

from dpcomp_core.algorithm import *
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload


# In[620]:

# Include visual_metric.py to calculate different error metric scores
import visual_metric
visual_metric


# In[621]:

# number of bins
# domain = (256,)
domain = 256

epsilon = 0.1

nickname = 'HEPTH'
# nickname = 'BIDS-ALL'

# number of data points from sample_to_scale data generation
sample = 1e4

# ESPairs = [(0.1, 1e4), (0.01, 1e5), (0.001, 1e6)]

seeds = range(20)

ds_seed = 111

# Instantiate dataset
data = dataset.DatasetSampledFromFile(nickname=nickname, 
                                     sample_to_scale=sample, 
                                     reduce_to_dom_shape=domain, 
                                     seed=ds_seed)

# Instantiate workload
# w = workload.Identity(domain_shape=domain)
w = workload.Prefix1D(domain_shape_int=domain)

# print w.__class__.__name__

# Instantiate algorithms
a = identity.identity_engine()
b = HB.HB_engine()
c = mwemND.mwemND_engine()
d = dawa.dawa_engine()

algorithms = [a, b, c, d]
# print identity.identity_engine().short_name


# In[622]:

# original data
dat = data.payload

df = pd.DataFrame(dat)
x = df.index.values
y = df.values.flatten()

# clrs = ['red' if (k == max(y)) else 'black' for k in y ]
# orig = sns.barplot(x=x, y=y, palette=clrs)

maxValOrig = max(y)
maxValIndexOrig = y.argmax()

print "Max: %d, Index: %d" % (maxValOrig, maxValIndexOrig)


# In[623]:

datCDF= np.cumsum(dat)


# In[631]:

errs = dict(dict())

# metr = ["M1", "M2", "M3"]
metr = ["L1", "L2", "Linf"]
# metr = ["L1", "L2", "Linf", "M1", "M2", "M3"]
# metr = ["chi-sq"]

for m in metr:
    errs[m] = {}
    for alg in algorithms:
        errs[m][alg.short_name] = []
# Use different seeds and average for a more accurate error sample
for seed in seeds:
    if seed % 10 == 0:
        print "SEED: ", seed
    for alg in algorithms:
#         print "Algorithm: ", alg.short_name
        
        x_hat = alg.Run(w, dat, epsilon, seed)
        df_hat = pd.DataFrame(x_hat)
        x_hat_data = df_hat.index.values
        y_hat_data = df_hat.values.flatten()

        # normalized non-negative rounding post-processing
        negSum = sum(y_hat_data)
        posSum = 0.00
        for i in y_hat_data:
            if i >= 0:
                posSum += i
        y_hat_data = [x*(negSum/posSum) if x >= 0 else 0 for x in y_hat_data]

        datCDF_noisy = np.cumsum(y_hat_data)
        
        settings = (epsilon, seed, ds_seed, alg, nickname, workload.Prefix1D, sample, domain)
        
        reload(visual_metric)
        vm = visual_metric.VisualMetric(y, y_hat_data, settings)
        for m in metr:
            errs[m][alg.short_name].append(vm.calcError(m))
            
# print errs

avgErrors = errs
for e in avgErrors:
    for alg in avgErrors[e]:
        v = errs[e][alg]
        if len(v) > 0:
            avgErrors[e][alg] = sum(avgErrors[e][alg])/len(avgErrors[e][alg])
        else:
            avgErrors[e][alg] = None
print avgErrors


# In[632]:

# Reformat as dataframe to graph as barplot
df = pd.DataFrame.from_dict(avgErrors)
df2 = df.stack()
df2 = df2.to_frame()
df3 = pd.DataFrame(df2.to_records())
df3.columns = ["alg", "metric", "error"]
df3


# In[633]:

g = sns.factorplot(x="metric", y="error", hue="alg", data=df3, kind="bar", palette="Blues", size = 8)
g.despine(left=True)
g.set_ylabels("Error (Prefix 1D workload)")
g.set_xlabels("Metric (varied units)")


# In[577]:

plotData = {}

for err in sorted(avgErrors):
    for i in sorted(avgErrors[err]):
        if avgErrors[err][i] != None:
            if err not in plotData:
                plotData[err] = [avgErrors[err][i]]
            else:
                plotData[err].append(avgErrors[err][i])
#         print err, i, avgErrors[err][i]
print plotData


# In[445]:

# Reformat to plot as barplot
f, ax = plt.subplots(figsize=(7, 7))
palettes = ["deep", "muted", "bright", "dark", "pastel"]
count = 0
for err in plotData:
    sns.set_color_codes(palettes[count])
#     print err, plotData[err]
    sns.barplot(x=["DAWA", "HB", "Identity", "MWEM"], y=plotData[err], label=err, color="g")
    count +=1
ax.legend(ncol=2, loc="upper right", frameon=True)
sns.despine(left=True, bottom=True)


# In[ ]:



