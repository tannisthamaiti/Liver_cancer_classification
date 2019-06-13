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
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
from numpy import genfromtxt
from wdfReader import *

def normalize(data):
    _min = np.min(data)
    _max = np.max(data)
    return (data - _min) / (_max - _min) 

class model_predict:
    def __init__(self, __file__= None, data= None, model_file = None, ml_model = None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path, model_file, ml_model), 'rb')
        self._rf_model = pickle.load(file)
        self.data= data
        file.close()
    def predict(self):
        #self._rf_model.predict(data= self.data)
            # converting the incoming dictionary into a numpy array that can be accepted by the scikit-learn model\n",
        X = normalize(self.data)
            # making the prediction and extracting the result from the array\n",
        y_hat = int(self._rf_model.predict(X)[0])
            #converting the prediction into a string that will match the output schema of the model\n",
            # this list will map the output of the scikit-learn model to the output string expected by the schema\n",
        targets = ['normal', 'presence of Type 1 Liver cancer', 'presence of Type 2 Liver cancer', 'presence of Type 3 Liver cancer', \
	'presence of Type 4 Liver cancer']
        precision = [89, 55, 34, 46, 55]
        species = targets[y_hat]
        dict_model = dict({"Type": species, "Label":y_hat, "Label":y_hat, 'Precision': precision[y_hat]})
        return (dict_model)
 
