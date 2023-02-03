import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, precision_score, recall_score
from pydataset import data
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import acquire
import prepare


def get_dum(df):   
    # Create dummies for non-binart categorical variables
    to_dummy = ['wine_color']
    dummies = pd.get_dummies(df[to_dummy], drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    
    # drop redundant column
    drop = ['wine_color']
    df.drop(columns=drop, inplace=True)


def get_baseline(df):
    df['baseline'] = df['quality'].value_counts().idxmax()
    (df['quality'] == df['baseline']).mean()
    # clean f string
    print(f"Baseline: {(df['quality'] == df['baseline']).mean()*100:.2f}%")
    

def get_rf(X_train, y_train):   
    rf2 = RandomForestClassifier(max_depth=3, random_state=42,
                            max_samples=0.5)
    #fit it 
    rf2.fit(X_train, y_train)
    # clean f string
    print('Random Forest Model')
    print(f"Accuracy of Random Forest on train data: {rf2.score(X_train, y_train)}") 
    print(f"Accuracy of Random Forest on validate: {rf2.score(X_validate, y_validate)}")
    
    
def get_logit(X_train, y_train):
    logit2 = LogisticRegression(C=.1, random_state=42, 
                           intercept_scaling=1, solver='newton-cg')

    #fit the model
    logit2.fit(X_train, y_train)
    #clean f string
    print('Logistic Regression Model')
    print(f"Accuracy of Logistic Regression on train: {logit2.score(X_train, y_train)}") 
    print(f"Accuracy of Logistic Regression on validate: {logit2.score(X_validate, y_validate)}")
    

def get_knn(X_train, y_train):
    knn3 = KNeighborsClassifier(n_neighbors=21)
    knn3.fit(X_train, y_train)
    knn3.score(X_train, y_train)
    knn3.score(X_validate, y_validate)
    # clean f string
    print('KNN Model')
    print(f"Accuracy of KNN on train: {knn3.score(X_train, y_train)}") 
    print(f"Accuracy of KNN on validate: {knn3.score(X_validate, y_validate)}")
    
    

def get_test(X_test, y_test):
    test_score = rf2.score(X_test, y_test)

    # clean f string
    print('Random Forest Model')
    print(f'Accuracy on Test {test_score:.2f}')