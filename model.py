# Feature Processing libraries

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import phik

#Algorithms libraries

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

##pipelines and transformers 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

##handling imbalance datasets
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_class_weight

##hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

##for hypothesis testing 

from scipy.stats import chi2_contingency

##model evaluation:
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

# Other packages
import os
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
import joblib
import gradio as gr
import pandas as pd 
import numpy as np 

# Modeling
model = joblib.load("models/LR.plk")


#Step 1: Data Splitting
##creating our features and label

df_drop= pd.read_csv("df_drop.csv")
X= df_drop.drop("Churn", axis=1)
y= df_drop.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

###converting our label to a numeric variable for easy analysis 

LE= LabelEncoder() ##initializing the model


num_y_train= LE.fit_transform(y_train) ##fitting and transforming on the train data

num_y_test= LE.transform(y_test) ##transforming on the test data

# Step 2: Creating Our Attributes


##getting our categorical attributes 
cat_attr= [i for i in df_drop.drop(["TotalCharges", "MonthlyCharges", "tenure", "Churn"], axis= 1)]


##getting our numerical attributes
num_attr= ["TotalCharges", "MonthlyCharges", "tenure"]

# Step 3: Creating Pipeline
from sklearn.preprocessing import MinMaxScaler

##This pipeline will handle the nan values in our dataset and also standardize our

## we are using mean because from our previous analysis, there were no outliers

num_pipeline= Pipeline([("mean_imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())])

cat_pipeline= Pipeline([("one_hot", OneHotEncoder())])

##we are combining our numeric and categorical pipelines with a Columntransformer

col_pipe= ColumnTransformer([("num_pipe", num_pipeline, num_attr),("cat_pipe", cat_pipeline, cat_attr)])

##initializing our class weight for each class

class_weights = compute_class_weight('balanced', classes=[0, 1], y=num_y_train)##initializing our class weight for each class
weight= dict(zip([0, 1], class_weights))


# Logistic Regressor Pipeline


LRP= Pipeline([##("spaceImputer", SpaceImputer(remove_space)),
               ("coltrans", col_pipe), 
              ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
              ("model", LogisticRegression(random_state= 100))
              ])

LRP.fit(X_train, num_y_train)

result_2= LRP.predict(X_test)
print(classification_report(num_y_test,result_2))

# Logistic Regression with Class Weight

CW_LRP= Pipeline([##("spaceImputer", SpaceImputer(remove_space)),
               ("coltrans", col_pipe),  
            ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
               ("model", LogisticRegression(
                   random_state=100,
                   class_weight=weight))
              
              ])
result_7=CW_LRP.predict(X_test)
print(classification_report(num_y_test, result_7))

# SMOTE with Logistic Regression
LGR_SM= Pipeline([##("spaceImputer", SpaceImputer(remove_space)),
               ("coltrans", col_pipe), 
               ("feature_selection", SelectKBest(score_func=f_classif, k=10)),# Perform feature selection
              ("smote", SMOTE(random_state=100)),  # Apply SMOTE for oversampling
               ("model", LogisticRegression(random_state= 100))  
              ])

LGR_SM.fit(X_train, num_y_train)
result_12= LGR_SM.predict(X_test)
print(classification_report(num_y_test, result_12))

# Bagging Logistic Regression with Class Weight

en_LRP= Pipeline([##("spaceImputer", SpaceImputer(remove_space)),
               ("coltrans", col_pipe),  
            ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
               ("model", BaggingClassifier(LogisticRegression(
                   random_state=100,
                   class_weight=weight), bootstrap_features= True,random_state= 100))
              
              ])

en_LRP.fit(X_train, num_y_train)

result_17= en_LRP.predict(X_test)
print(classification_report(num_y_test, result_17))

# Hyperparameter Tuning
##loading our model 

CW_LRP

##creating a set of C-values for our hyperparameter to loop over to see which one gives us a lower log loss

#np.geomspace(1e-5, 1e5, num=20) 

## geomspace generates a set of evenly spaced numbers over a logarithimic scale

##plotting our np.geomspace(1e-5, 1e5, num=20) to see if it is evenly spaced out 

##this will help us visualize the spacing logarithmicly
#plt.plot(np.geomspace(1e-5, 1e5, num=20)) 

##this will help us visualize the spacing linearly since the logarithmic method can be difficult

#plt.plot(np.linspace(1e-5, 1e5, num=20)) 

#plt.title("Linear and Logarithmic Visualization of the Chosen C-Values")

#plt.show()

##We are going to create a copy of the CW_LRP and rename it for the purpose of hyperparameter tuning

HP_LRP= Pipeline([##("spaceImputer", SpaceImputer(remove_space)),
               ("coltrans", col_pipe),  
            ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
               ("model", LogisticRegression(
                   random_state=100,
                   class_weight=weight))
              
              ])#


###creating the params

params = {
    "model__C": np.geomspace(1e-5, 1e5, num=20),
    "model__max_iter": [100, 1200],
    "model__random_state": [24, 42, 57, 100]
}

##setting our parametersand telling it to return the model with the lowest log loss
##the scorer is setting log loss as our metric, and also, I am telling it to choose the model with the lowest score
## the need_proba is telling it to accept returned values and calculate the probability of each predicted label 

scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

Grid_HPT= GridSearchCV(estimator=HP_LRP, param_grid=params,scoring=scorer, cv=5)

Grid_HPT.fit(X_train, num_y_train)

##saving the best parameters and scores into the following variables

best_params = Grid_HPT.best_params_
best_score = Grid_HPT.best_score_
cv_results = Grid_HPT.cv_results_

print("Best Parameters:", best_params)
print("Best Parameters:", best_score)

##setting the parameters gotten from the hyperparameter tuning

HP_LRP.set_params(
                  model__random_state= 24,
                  model__max_iter=100,
                  model__C= 0.1623776739188721)

##fitting our pipeline with the parameters to our train and test variables

HP_LRP.fit(X_train, num_y_train)

# Model Evaluation For Logistic Regression

##loading my Logistic Regression Pipeline

CW_LRP

##we are predicting over  our X_train to get values for our log loss function

LR_train_pred= CW_LRP.predict(X_train)
LR_train_pred_proba= CW_LRP.predict_proba(X_train)

##note, log-loss accepts the predicted probability for the predicted value, and not the actual predicted value

print ("the log loss of the training set is for untuned Logistic Regression is: ",  log_loss(num_y_train,LR_train_pred_proba))

##Let's ge the probabilityloss of the test set:

LR_test_pred_proba= CW_LRP.predict_proba(X_test)
print ("the log loss of the test set is for untuned Logistic Regression is: ",  log_loss(num_y_test,LR_test_pred_proba))

# Model evaluation of the Tuned Logistic Regression Model

HP_LR_train_pred_proba= HP_LRP.predict_proba(X_train)
print ("the log loss of the train set is for tuned Logistic Regression is: ",  log_loss(num_y_train,HP_LR_train_pred_proba))

HP_test_pred_proba= HP_LRP.predict_proba(X_test)
print ("the log loss of the test set is for the tuned Logistic Regression is: ",  log_loss(num_y_test,HP_test_pred_proba))

## Classification Reports for Both Tunes and Untuned Model

tuned_result= HP_LRP.predict(X_test)
untuned_result= CW_LRP.predict(X_test)
print("Classification report for Tuned LR is:  \n\n", classification_report(num_y_test, tuned_result))

print("Classification report for unTuned LR is\n\n", classification_report(num_y_test, untuned_result))

#df_copy

# Deployment

#X_test.to_csv("datahub/LP3/test_telco.csv")
HP_LRP

from joblib import dump, load
dump(HP_LRP, "LR.plk")

classifier = joblib.load('C:\\Users\\otchi\\data_analytics\\virenv\\LR.plk')
#classifier
#classifier.predict(X_test)
