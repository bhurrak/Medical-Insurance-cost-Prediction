# Medical-Insurance-cost-Prediction
import pandas as pd
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=load_boston()
df
dataset=pd.DataFrame(df.data)
dataset
dataset.columns=df.feature_names
dataset.head()
## Independent features and dependent features
X=dataset
y=df.target
y
## train test split 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)
X_train
## standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
##cross validation
from sklearn.model_selection import cross_val_score
regression=LinearRegression()
regression.fit(X_train,y_train)
mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
np.mean(mse)
##prediction 
reg_pred=regression.predict(X_test)
reg_pred
import seaborn as sns
sns.displot(reg_pred-y_test,kind='kde')
from sklearn.metrics import r2_score
score=r2_score(reg_pred,y_test)
score
