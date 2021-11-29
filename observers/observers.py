# observers.py
# observers of model

# dependencies
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import kurtosis, ttest_1samp


# auto correlation
# calculate the acf for MC simulation result, each row is a single simulation result
def acf_multiple(x, nlags=100):
    results = [acf(x[i, :], nlags=nlags) for i in range(x.shape[0])]
    return np.vstack(results)

# calculate the mean and sd of acfs of MC simulation result
def acf_mean_sd(x, nlags=100):
    acfs = acf_multiple(x, nlags=nlags)
    acfs_mean = np.mean(acfs, axis=0)
    acfs_sd = np.std(acfs, axis=0, ddof=1)
    
    return acfs_mean, acfs_sd


# distribution shape
# calculate the kurtosis for MC simulation results, Pearson's definition(normal -> 3.0)
def kurtosis_multiple(x):
    return kurtosis(x, axis=1, fisher=False)

# kurtosis t test
# alternative: two-sided, less, greater
def kurtosis_t_test(x, popmean=3.0, alternative='greater'):
    krus = kurtosis_multiple(x)

    return ttest_1samp(krus, popmean=popmean, alternative=alternative), krus

# volatility clustering
