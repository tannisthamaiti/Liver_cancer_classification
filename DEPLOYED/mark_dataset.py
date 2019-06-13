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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
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

class1 = [1200, 1250, 1340]
class0 = [1210, 1330, 1400]
class2 = [1250, 1310, 1390]
class3 = [1210, 1350, 1380]
class4 = [1200, 1270, 1360]
peak_string = ['peak1', 'peak2', 'peak3']
class_label =dict({'0': class0, '1': class1, '2': class2, '3': class3, '4': class4})

class Dataset_marker:
    def __init__(self, xval=None, yval=None, xvalnorm= None, yvalnorm = None, pathname = None, filename = None, type_class=None):
        self.xval = xval
        self.yval = yval
        self.xvalnorm = xvalnorm
        self.yvalnorm = yvalnorm
        self.pathname =  pathname
        self.filename = filename
        self.type_class = type_class
    def plotthing_marker(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(self.xval, savgol_filter(normalize(self.yval),103,4),'k', label= 'Uploaded spectrum', linewidth=5)
        print(self.type_class)
        a=class_label[self.type_class]
        for i in range(3):
                #ax.text(class_type[i], 0.8, peak_string[i], fontsize=16)
        	ax.axvline(a[i],ymin=0.1, ymax=1.0,linewidth=4, color='r', linestyle ='--')
        #ax.plot(self.xvalnorm, normalize(self.yvalnorm),'r', label= 'Normal spectrum')
        ax.set_xlabel('Raman shift (cm-1)', fontsize=24)
        ax.set_ylabel('Intensity', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        legend = ax.legend(loc='upper center', shadow=False, fontsize=20)
        #legend.get_frame().set_facecolor('C0')
        plt.savefig(os.path.join(self.pathname, self.filename))


		
#print(read_spectrum('disease.wdf'))
