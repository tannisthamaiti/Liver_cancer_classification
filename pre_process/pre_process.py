'''
Class dealing with preprocessing the Raman data
'''
import sys
import seaborn as sns
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from imblearn.over_sampling import SMOTE
import random
import os
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from convertwdf import *
from wdfReader import * 
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
from itertools import product
from functools import partial, update_wrapper
from sklearn.metrics import accuracy_score
from scipy.signal import savgol_filter
sys.path.append("../")
import numpy as np
import scipy
from matplotlib import pyplot as plt
import detect_peaks
import rampy 
from sklearn import preprocessing
from detect_peaks import detect_peaks

def detect_peaks_filter(df_spectrum):
    peaks_total = []
    for i in range(len(df_spectrum)):
        x = df_spectrum.iloc[i]
        x_s = savgol_filter(x, 103, 4)
        peaks_total.append(Xspectra[detect_peaks(x_s, mph=1e-3, mpd=0, threshold=0.01, show=False)])
    return(peaks_total)
def find_labels(df_spectrum, labels, number):
    indices = (labels == number)
    histdata = df_spectrum.loc[indices]
    histdata = histdata.fillna(0.0)
    return(histdata)
def baseline_removal(df_spectrum):
    als_total = []
    # need to define some fitting regions for the spline
    roi = np.array([[0,100],[200,220],[280, 290],[420,430],[480,500]])
    # background: a large gaussian + linear 
    x = np.linspace(50, 1400, 1015)
    bkg = (60.0 * np.exp(-np.log(2) * ((x-250.0)/200.0)**2) + 0.1*x)*0.001
    for i in range(len(df_spectrum)):
        ycalc_als, base_als = rampy.baseline(x,df_spectrum.iloc[i],roi,'als',lam=10**7,p=0.05)
        B = np.asarray(ycalc_als)
        als_total.append(B)
    return(als_total)   

total_spectrum = pd.read_csv('Spectrum.csv')
labels_spectrum = pd.read_csv('Spectrum_labels.csv')
Xspectra = np.loadtxt('Spectrumnumber.dat')
feature_columns =total_spectrum.columns
test_1 = total_spectrum[feature_columns[1:]]
test_1.fillna(0.0, inplace=True)
Spectrum_feature = pd.DataFrame(detect_peaks_filter(test_1))
Spectrum_feature.to_csv('Spectrum_features_savgol.csv')
Spectrum_feature.fillna(0.0, inplace=True)
A = baseline_removal(test_1)
x = np.reshape(A, (len(A), 1015))
Spectrum_baseline = pd.DataFrame(x)
Spectrum_baseline.fillna(0.0, inplace=True)
Spectrum_baseline.to_csv('Spectrum_baseline.csv')