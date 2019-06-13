#!/usr/bin/python3
from dataset import *
from filter_dataset import *
from model_predict import * 
from mark_dataset import *
import numpy as np
from wdfReader import * 

def func(filename):
	wdfIle = wdfReader('normal.wdf')
	X = wdfIle.get_xdata()
	data = wdfIle.get_spectra()
	wdfIle = wdfReader(filename)
	X1 = wdfIle.get_xdata()
	data1 = wdfIle.get_spectra()
	model_cnn = model_predict('/home/titli/Documents/SpectralRaman/Livercancer', data1.reshape(1,1015), 'Livercancer/model_file', 'model_rf.pkl')
	a =model_cnn.predict()
	xyz = Dataset(X, data, X1, data1, '/home/titli/Documents/SpectralRaman/Livercancer/static/img', '2.png')
	xyz.plotthing()
	xyz1 = Dataset_filter(X, data, X1, data1, '/home/titli/Documents/SpectralRaman/Livercancer/static/img', '1.png')
	xyz1.plotthing_filter()
	xyz2 = Dataset_marker(X, data, X1, data1, '/home/titli/Documents/SpectralRaman/Livercancer/static/img', '3.png', str(a['Label']))
	xyz2.plotthing_marker()
	return(model_cnn.predict())


#from flask import Flask
#import your_module # this will be your file name; minus the `.py`

#app = Flask(__name__)

#@app.route('/')
#def dynamic_page():
#    return your_module.your_function_in_the_module()
#
#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port='8000', debug=True)
