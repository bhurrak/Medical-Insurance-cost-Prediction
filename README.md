# Medical-Insurance-cost-Prediction

Medical Insurance Cost Prediction using Linear Regression
This project uses a simple linear regression model to predict medical insurance costs based on various features. The model is built and evaluated in Python using scikit-learn and the Boston housing dataset for demonstration.

Table of Contents
Project Overview
Dataset
Dependencies
Installation
Project Structure
Model Training
Evaluation
Results
License
Project Overview
This project demonstrates the practical implementation of linear regression using Python and the scikit-learn library. The goal is to illustrate how to build, train, and evaluate a simple linear regression model using a well-known dataset.

Dataset
The Boston Housing dataset is used in this project, containing features such as:

CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built before 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property-tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk - 0.63)^2, where Bk is the proportion of Black residents by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s (target variable)
Dependencies
The following libraries are required to run the project:

Python 
Pandas
NumPy
Scikit-learn
Matplotlib

Model Training
The following steps are included in the model training process:

Data Preprocessing: Loading and preparing the dataset.
Exploratory Data Analysis (EDA): Visualizing the data to identify relationships.
Model Training: Fitting a linear regression model.
Prediction: Using the model to predict the target variable.

Evaluation
The model was evaluated using metrics such as Mean Squared Error (MSE) and R-Squared (R²). These metrics help measure the accuracy and reliability of the predictions.

The model achieved an R² score of 0.669 and  on the test set. These results indicate that the linear regression model can provide a reasonably accurate prediction for the target variable.

