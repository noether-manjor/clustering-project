import pandas as pd
import numpy as np
import os


def get_red_wine():
    '''this function reads in a csv file and returns a dataframe'''
    
    filename = "winequality-red.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
def get_white_wine():
    '''this function reads in a csv file and returns a dataframe'''
    
    filename = "winequality-white.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
       

