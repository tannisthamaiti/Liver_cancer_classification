# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request, redirect, url_for
from numpy import genfromtxt
from werkzeug import secure_filename
import pickle
import os
from wdfReader import * 
from call_plot_functions import func
import time

# Create the application object

UPLOAD_FOLDER = '/home/titli/Documents/SpectralRaman/readfile'
ALLOWED_EXTENSIONS = set(['csv', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wdf'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#@app.route('/',methods=["GET","POST"])
#def home_page():
#    return render_template('upload.html')  # render a template
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def start():
    return render_template('upload.html')




@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text = func(filename)
        time.sleep(5)
        print(text)
    return render_template('result.html', figure='2.png', int_var=text['Type'], per_var = text['Precision'])



#########################################################################


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/

