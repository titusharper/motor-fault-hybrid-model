import os
import numpy as np
import pandas as pd
import scipy.io
import glob

EPS = 1e-12
def safe_std(x):
    s = np.std(x)
    return s if s > EPS else EPS
def safe_mean_abs(x):
    m = np.mean(np.abs(x))
    return m if m > EPS else EPS
def safe_smr(x):
    # square mean root = (mean(sqrt(|x|)))**2
    smr = (np.mean(np.sqrt(np.abs(x))))**2
    return smr if smr > EPS else EPS

def extract_features(x):
    # 19 features, x: 1D numpy array (float)
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    mu = np.mean(x)
    sd = safe_std(x)
    abs_mean = safe_mean_abs(x)
    smr = safe_smr(x)
    rms = np.sqrt(np.mean(x**2) + EPS)
    maxabs = np.max(np.abs(x))
    z = (x - mu) / sd

    f = {}
    f['peak']               = maxabs
    f['rms']                = rms
    f['kurtosis']           = np.mean(z**4)
    f['crest_factor']       = maxabs / rms
    f['clearance_factor']   = (maxabs / np.mean(np.sqrt(np.abs(x)) + EPS))**2
    f['impulse_factor']     = maxabs / abs_mean
    f['shape_factor']       = rms / abs_mean
    f['skewness']           = np.mean(z**3)
    f['square_mean_root']   = smr
    f['moment_5']           = np.mean(z**5)
    f['moment_6']           = np.mean(z**6)
    f['mean']               = mu
    f['shape_factor2']      = smr / abs_mean
    f['peak_to_peak']       = np.max(x) - np.min(x)
    f['kurtosis_factor']    = f['kurtosis'] / ((np.mean(x**2) + EPS)**2)
    f['std']                = sd
    f['smoothness']         = 1 - np.sqrt(1 + sd**2)
    f['uniformity']         = 1 - sd / (mu if abs(mu) > EPS else (np.sign(mu)*EPS or EPS))
    f['normal_neg_loglike'] = (1.0 / (sd*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2) / (2*sd**2))
    f['normal_neg_loglike'] = float(np.mean(f['normal_neg_loglike']))
    return [f[k] for k in f]
