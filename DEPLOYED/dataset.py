#!/usr/bin/python3
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from imblearn.over_sampling import SMOTE
import random
import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
from numpy import genfromtxt
from wdfReader import * 
from scipy.signal import savgol_filter



font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 14,
                }
def normalize(data):
    _min = np.min(data)
    _max = np.max(data)
    return (data - _min) / (_max - _min)

class Dataset:
    def __init__(self, xval=None, yval=None, xvalnorm= None, yvalnorm = None, pathname = None, filename = None):
        self.xval = xval
        self.yval = yval
        self.xvalnorm = xvalnorm
        self.yvalnorm = yvalnorm
        self.pathname =  pathname
        self.filename = filename
    def plotthing(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(self.xval, savgol_filter(normalize(self.yval), 103, 4),'b', label= 'Uploaded spectrum')
        ax.plot(self.xvalnorm, savgol_filter(normalize(self.yvalnorm), 103, 4),'r', label= 'Normal spectrum')
        ax.set_xlabel('Raman shift (cm-1)', fontsize=24)
        ax.set_ylabel('Intensity', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        legend = ax.legend(loc='upper center', shadow=False, fontsize=20)
        #legend.get_frame().set_facecolor('C0')
        plt.savefig(os.path.join(self.pathname, self.filename))


		
#print(read_spectrum('disease.wdf'))
