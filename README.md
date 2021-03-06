# LIVE-r
Please find demo of app here
[![LIVE-r](../master/img/LIVE-r.png)](https://www.youtube.com/watch?v=ozPiA5uE3Kw&feature=youtu.be "LIVE-r")
Please find link to presention ![here](../master/LIVE-r%20Demo.pdf) 

## Liver cancer detection and classification system
A web application that can be used by clinicians to detect presence of liver cancer. The input parameter to the app is Raman spectrum files that are in wdf format. The model is based on a machine learning algorithm that is trained on 10000 blood samples which gives an accuracy of 97%.
The methodology to build predictive models on five classes of skewed data by selecting Raman spectrum data of blood samples. This methodology can be applied to any domain to generate predictive system on skewed datasets and many features.

## Dev Environment
Run the requirement.txt file to install all the dependencies.

## File Structure

* `binary_spectrum.py, multiclass_spectrum.py, multiclass_pca_spectrum.py`: main file.
* `pre_process.py`: script used to pre_process spectrum samples.
* `Deployment`: Folder deployed to AWS.
## Architecture Diagram
![Optional Text](../master/img/archi.png)
1.Spectral data are pre-processed with filter and baseline correction.  
2.Saved as csv file.  
3.Run machine learning models for classification.    
4.Deploy to amazon web app.
## Steps
1.Pre-processing Data  
2. Imbalanced data set  
3. Xgboost  
4. Examples of use  
5. Evaluate model  
## 1. Pre-processing Data
The Preprocessing notebook has a full spectral preprocessing tutorial using filter and baseline correction. The figure defines how a baseline was choosen and was subtracted from spectrum data.
![Optional Text](../master/img/baseline.png)

## 2. Imbalanced data set
When faced with imbalanced data sets there is no one stop solution to improve the accuracy of the prediction model. In most cases, synthetic techniques like SMOTE will outperform the conventional oversampling and undersampling methods. In this pattern, we can see the changes in the output with different runs using different techniques and users can play around a bit with the parameters tuning to arrive at optimum results. This is an attempt to demonstrate the methodology to handle skewed data and generate predictive models.
## 3. Xgboost
Depending on the system configueration, we can select the Bagging or Boosting Algorithm. Bagging improves accuracy of machine learning algorithms by creating aggregated models with less variance. Boosting is an ensemble technique which emphasizes on training for weak learners to create a strong learner that can make accurate predictions.
## 4. Example of Use
Python file `multiclass_spectrum.py` gives results for recall and precision for multiclass classificiation. 
## 5. Evaluate model
| Class        | Precision (%)           | Recall (%) |
| ------------- |:-------------:| -----:|
| Normal    | 90 | 80 |
| Disease 1 | 71 | 34 |
| Disease 2 | 63 | 88 |
| Disease 3 | 70 | 56 |
| Disease 4 | 75 | 51 |
