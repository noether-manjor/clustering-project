import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import acquire
import os

def prep_wine(red_wine, white_wine):
    '''this function preps the wine csv's, concatenates them, changes data types, 
        and renames columns'''
    # take in the df's and concant and reset index
    red_wine.insert(12, column="wine_color", value='red')
    white_wine.insert(12, column="wine_color", value='white')
    frames = [red_wine, white_wine]
    wines = pd.concat(frames)
    wines = wines.reset_index(drop=True)
    
    # change some data types
    wines["free sulfur dioxide"] = wines["free sulfur dioxide"].astype(int)
    wines["total sulfur dioxide"] = wines["total sulfur dioxide"].astype(int)
    
    # rename some columns
    wines = wines.rename(columns={'fixed acidity':'fixed_acidity', 
                            'volatile acidity':'volatile_acidity', 
                            'citric acid':'citric_acid',
                            'residual sugar':'residual_sugar', 
                            'free sulfur dioxide':'free_sulfur_dioxide',
                            'total sulfur dioxide':'total_sulfur_dioxide'
                            })
    
    # Create dummies for non-binart categorical variables
    to_dummy = ['wine_color']
    dummies = pd.get_dummies(wines[to_dummy], drop_first=False)
    wines = pd.concat([wines, dummies], axis=1)
    
    # drop redundant column
    drop = ['wine_color']
    wines.drop(columns=drop, inplace=True)
    # get rid of outliers
    col_list = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                            'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
                            'alcohol']
    for col in col_list:

        q1, q3 = wines[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        k=1.5
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        wines = wines[(wines[col] > lower_bound) & (wines[col] < upper_bound)]
        
    return wines

def split_data(df, target):
    '''
    This function take in a dataframe performs a train, validate, test split
    Returns train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test
    and prints out the shape of train, validate, test
    '''
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)

    #Split into X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    # Have function print datasets shape
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
   
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test