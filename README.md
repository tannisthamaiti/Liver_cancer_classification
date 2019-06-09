# LIVE-r
## Liver cancer detection and classification system

The methodology to build predictive models on five classes of skewed data by selecting Raman spectrum data of blood samples. This methodology can be applied to any domain to generate predictive system on skewed datasets and many features.

## Dev Environment
Run the requirement.txt file to install all the dependencies.

## File Structure

* `binary_spectrum.py, multiclass_spectrum.py, multiclass_pca_spectrum.py`: main file.
* `pre_process.py`: script used to pre_process spectrum samples.
* `Deployment`: Folder deployed to AWS.
## Architecture Diagram
![Optional Text](../master/img/archi.png)
## Steps
1.Pre-processing Data  
2. Imbalanced data set  
3. Xgboost  
4. Examples of use  
5. Evaluate model  
## Pre-processing Data
![Optional Text](../master/img/baseline.png)

## Imbalanced data set
When faced with imbalanced data sets there is no one stop solution to improve the accuracy of the prediction model. In most cases, synthetic techniques like SMOTE will outperform the conventional oversampling and undersampling methods. In this pattern, we can see the changes in the output with different runs using different techniques and users can play around a bit with the parameters tuning to arrive at optimum results. This is an attempt to demonstrate the methodology to handle skewed data and generate predictive models.
## Xgboost
Depending on the system configueration, we can select the Bagging or Boosting Algorithm. Bagging improves accuracy of machine learning algorithms by creating aggregated models with less variance. Boosting is an ensemble technique which emphasizes on training for weak learners to create a strong learner that can make accurate predictions.
