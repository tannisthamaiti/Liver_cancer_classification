#!/usr/bin/python3
import numpy as np
import pandas as pd
import sys
from wdfReader import *

def convert(path, **kwarg):
	if path[-3:]=='wdf':
		try:
			data = wdfReader(path)
		except:
			print("Could not read the file")
			return
		X = data.get_xdata()
		Y = data.get_spectra()
		nr = data.count
		header = 'Number of spectra: %d\nPoints per spectrum: %d\nAccumulation count: %d\nLaser wavenumber: %.6f\n' %(data.count, data.point_per_spectrum, data.accumulation_count, data.laser_wavenumber)
		if nr ==1:
			print("File contains one spectrum")
			dt = np.column_stack((X, Y))
			np.savetxt(path[:-3]+'txt', dt, delimiter='	', header=header)
			print("Text file was exported")
		else:
			print("File contains %d spectra" %nr)
			out = pd.DataFrame()
			out['X'] = X
			for i in np.arange(0, nr):
				out[str(i+1)]=Y[i*len(X):(i+1)*len(X)]
			out.to_csv(path[:-3]+'csv', index=None)
			print("CSV file exported")
			if 'text' in kwarg:
				np.savetxt(path[:-3]+'txt', out.values[:,:2], header=header, delimiter='	')
if len(sys.argv)>1:
	convert(sys.argv[1])
