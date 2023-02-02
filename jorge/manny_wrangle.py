import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from env import get_db_url

from scipy import stats
from sklearn.model_selection import train_test_split

import sklearn.preprocessing
from IPython.display import display, Markdown

#                             _              _        
#                          __| | _   _  ___ | |_  ___ 
#                         / _` || | | |/ __|| __|/ __|
#                        | (_| || |_| |\__ \| |_ \__ \
#                         \__,_| \__,_||___/ \__||___/
                            
def get_data(sql_db, query):
    '''
        Accepts 2 arguments of string type:
        1: SQL database name
        2: SQL query
        
        Checks if .csv already exists before
        connecting with SQL database again
        
        Saves a .csv file of DataFrame
        
        Returns DataFrame
    '''
    
    import os
    import pandas as pd
    
    # variable to hold filename created from 
    # input argument of SQL database name
    path = f'{sql_db}.csv'
    
    # Holds boolean result of check for
    # .csv existing; uses OS module
    file_exists = os.path.exists(path)
    
    # Uses boolean value variable to
    # check whether to create a new
    # SQL connection or load .csv
    #
    # Finished off by returning DataFrame
    if file_exists:
        print('Reading CSV')
        df = pd.read_csv(path)
        
        return df

    else:
        print('Downloading SQL DB')

        url = get_db_url(sql_db)
        df = pd.read_sql(query, url)
        df.to_csv(f'{sql_db}.csv',index=False)
        return df

def clean_data(df):
    '''
        Accepts DataFrame from get_data() function in wrangle.py
        H A R D C O D E D   O P E R A T I O N S
            &
        Returns a cleaned DataFrame
    '''
    
    # Drop Null Columns
    # =================

    # List of Null Column Values
    df.isnull().sum().sort_values(ascending=False)[:34]

    # List of Null Column Names
    drop_column_list = ["buildingclasstypeid"\
                        , "finishedsquarefeet15"\
                        , "finishedsquarefeet13"\
                        , "storytypeid"\
                        , "basementsqft"\
                        , "yardbuildingsqft26"\
                        , "architecturalstyletypeid"\
                        , "typeconstructiontypeid"\
                        , "fireplaceflag"\
                        , "finishedsquarefeet6"\
                        , "decktypeid"\
                        , "pooltypeid10"\
                        , "poolsizesum"\
                        , "pooltypeid2"\
                        , "hashottuborspa"\
                        , "yardbuildingsqft17"\
                        , "taxdelinquencyflag"\
                        , "taxdelinquencyyear"\
                        , "finishedfloor1squarefeet"\
                        , "finishedsquarefeet50"\
                        , "threequarterbathnbr"\
                        , "fireplacecnt"\
                        , "pooltypeid7"\
                        , "poolcnt"\
                        , "airconditioningtypeid"\
                        , "numberofstories"\
                        , "garagetotalsqft"\
                        , "garagecarcnt"\
                        , "regionidneighborhood"\
                        , "buildingqualitytypeid"\
                        , "unitcnt"\
                        , "propertyzoningdesc"\
                        , "heatingorsystemtypeid"]
                        
    # Dropping Columns using 'drop_column_list'
    df.drop(columns = drop_column_list, inplace=True)


    # Drop Null Rows
    # ==============

    # List of Null Rows Values
    less_than_1100_nulls = df.isnull().sum().sort_values(ascending=False)[:13]

    # List of Null Rows Names
    less_than_1100_nulls_list = ['regionidcity'\
                            ,'lotsizesquarefeet'\
                            ,'finishedsquarefeet12'\
                           ,'calculatedbathnbr'\
                           ,'fullbathcnt'\
                           ,'censustractandblock'\
                           ,'yearbuilt'\
                           ,'structuretaxvaluedollarcnt'\
                           ,'calculatedfinishedsquarefeet'\
                           ,'regionidzip'\
                           ,'taxamount'\
                           ,'landtaxvaluedollarcnt'\
                           ,'taxvaluedollarcnt']
                           
    # Drop Null Rows using subset and list of names
    df = df.dropna(subset=less_than_1100_nulls_list)


    #Fix Data Types
    # ==============

    # temporarily converting to interger to remove
    # trailing zeroes
    df['fips'] = df['fips'].apply(int).copy()

    # converting to final datatype as string
    df['fips'] = df['fips'].apply(str).copy()

    # as string adding a leading '0'
    df['fips'] = '0' + df['fips'].copy()

    # convert 'yearbuilt' to interger
    df['yearbuilt'] = df['yearbuilt'].apply(int).copy()

    # convert 'assessmentyear' to interger
    df['assessmentyear'] = df['assessmentyear'].apply(int).copy()


    # Rename Columns
    # ==============

    # list of new names
    new_column_names = ['parcelid'\
    ,'id'\
    ,'bathroom_count'\
    ,'bedroom_count'\
    ,'calculated_bathandbr'\
    ,'calculated_finished_square_feet'\
    ,'finished_square_feet_12'\
    ,'fips'\
    ,'full_bath_count'\
    ,'latitude'\
    ,'longitude'\
    ,'lot_size_square_feet'\
    ,'property_county_landuse_code'\
    ,'property_land_use_type_id'\
    ,'raw_census_tract_and_block'\
    ,'region_id_city'\
    ,'region_id_county'\
    ,'region_id_zip'\
    ,'room_count'\
    ,'year_built'\
    ,'structure_taxvalue_dollarcount'\
    ,'tax_valuedollar_count'\
    ,'assessment_year'\
    ,'land_tax_value_dollar_count'\
    ,'tax_amount'\
    ,'census_tract_and_block'\
    ,'id_1'\
    ,'log_error'\
    ,'transaction_date']

    # renaming
    df.set_axis(new_column_names, axis=1,inplace=True)

    # feature engineering
    df.drop(columns='calculated_bathandbr',inplace=True)
    df['bed_bath_count']=df['bathroom_count']+df['bedroom_count']
    
    return df

def split_data(df,target=None,seed=42):    
    '''
        Accepts 2 arguments;
        DataFrame, and Keyword Value for 'stratify'
        ...
        Splits DataFrame into a train, validate, and test set
        and it will return three DataFrames stratified on target:
        
        train, val, test (in this order) -- all pandas Dataframes
        60%,20%,10% 
    '''
    
    train, test = train_test_split(df,
                               train_size = 0.8, random_state=seed)
    train, val = train_test_split(df,
                             train_size = 0.75,
                             random_state=seed)
    return train, val, test

def isolate_target(df,target=None):
    '''
        Isolates Target Variable from DataFrame
        Returns X & y DataFrames
    '''
    X = df.drop(columns=target)
    y = df[target]
    return X,y

def wrangle_zillow():
    '''
        Main function in `wrangle.py`
        When run, wrangle_zillow will utilize
        get_db_url(), get_data(), and clean_data()
        
        to acquire & prepare DataFrame
        
        returns a DataFrame
    '''
    
    sql_db = "zillow"
    query = "SELECT * FROM properties_2017 JOIN predictions_2017 USING(parcelid) WHERE (`propertylandusetypeid` = 261) & (YEAR(`transactiondate`) = 2017);"
    df = get_data(sql_db,query)
    df = clean_data(df)
    
    return df

def wrangle_mall():
    '''
        Main function in `wrangle.py`
        When run, wrangle_mall will utilize
        get_db_url(), get_data(), and clean_data()
        
        to acquire & prepare DataFrame
        
        returns a DataFrame
    '''
    
    sql_db = "mall_customers"
    query = '''
            SELECT *
            FROM customers
            '''
    df = get_data(sql_db,query)
    #df = clean_data(df)
    
    return df

def scale_MinMaxScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Min Max Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def scale_StandardScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Standard Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def scale_RobustScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Robust Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.RobustScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def scale_data(df,scaler=None):
    '''
        All-In-One Scaler Function.
        Seperates Continous Columns, and Default
        Uses MinMax Scaler.
    '''
    num, _ = separate_column_type_list(df)
    num = df[num]
    
    if scaler == 'RobustScaler':
        return RobustScaler(num)
    
    elif scaler == 'StandardScaler':
        return StandardScaler(num)
    
    else:
        return MinMaxScaler(num)

def compare_scalers(df):
    '''
        Accepts a DataFrame
        
        Is used to visualize 3 scaler outputs
        and compare to original DataFrame
    '''
    mm_scaled = MinMaxScaler(df)
    ss_scaled = StandardScaler(df)
    rs_scaled = RobustScaler(df)

    font = {'family': 'Georgia',
            'color':  '#525252',
            'weight': 'bold',
            'size': 25,
            }
    # ====================================================================

    # Assigning 'fig', 'ax' variables.
    fig, ax = plt.subplots(2, 2,figsize=(25,25))

    # Defining custom 'xlim' and 'ylim' values.
    custom_xlim = (0, 8000)

    # Setting the values for all axes.
    #plt.setp(ax, xlim=custom_xlim)
    
    # ====================================================================
    
    # Original Data
    ax[0][0].hist(df, color="#525252",ec='white',bins=10000)
    ax[0][0].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[0][0].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[0][0].set_title("Original",color='#525252', fontdict=font)
    ax[0][0].set_xlim([0, 8000])
    
    
    # MinMax Scaled
    ax[0][1].hist(mm_scaled, color="#525252",ec='white',bins=10000)
    ax[0][1].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[0][1].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[0][1].set_title("MinMax Scaled",color='#525252', fontdict=font)
    ax[0][1].set_xlim([0, .005])
    
    # Standard Scaled
    ax[1][0].hist(ss_scaled, color="#525252",ec='white',bins=10000)
    ax[1][0].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[1][0].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[1][0].set_title("Standard Scaled",color='#525252', fontdict=font)
    ax[1][0].set_xlim([-1.5, 3])

    # Robust Scaled
    ax[1][1].hist(rs_scaled, color="#525252",ec='white',bins=10000)
    ax[1][1].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[1][1].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[1][1].set_title("Robust Scaled",color='#525252', fontdict=font)
    ax[1][1].set_xlim([-2, 5])

def get_rmse(value,pred):
    '''
        Returns RMSE using actual values
        and predicted value
    '''
    return mean_squared_error(value,pred)**(1/2)

def rfe(X,y,k=2):
    '''
        Automates using RFE().
        Accepts 3 arguments:
        Provide X,y DataFrames with
        number of desired features.
    '''
    olm = LinearRegression()
    rfe = RFE(olm,n_features_to_select=k)
    rfe.fit(X,y)
    
    mask = rfe.support_
    
    return X.columns[mask]

def select_kbest(X,y,k=2):
    '''
        Automates using SelectKBest().
        Accepts 3 arguments:
        Provide X,y dataframes with
        number of desired features.
    '''
    f_selector = SelectKBest(f_regression,k=k)
    f_selector.fit(X,y)
    mask = f_selector.get_support()
    return X.columns[mask]

def eval_results(p, alpha, group1, group2):
    '''
        Test Hypothesis  using Statistics Test Output.
        This function will take in the p-value, alpha, and a name for the 2 variables
        you are comparing (group1 and group2) and return a string stating 
        whether or not there exists a relationship between the 2 groups. 
    '''
    if p < alpha:
        return f'There exists some relationship between {group1} and {group2}. (p-value: {p:.4f})'
    else:
        return f'There is not a significant relationship between {group1} and {group2}. (p-value: {p:.4f})'

def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
        Drops columns and rows in a Pandas DataFrame if they have a proportion
        of null values less than the required cutoffs.
        Parameters
        ----------
        df : Pandas DataFrame
            The DataFrame from which null values are to be dropped.
        prop_required_column : float
            The proportion of null values required for a column to be dropped.
        row_cutoff : float
            The proportion of null values required for a row to be dropped.

        Returns
        ----------
        df : Pandas DataFrame
            The DataFrame with columns and rows with null values dropped.
    '''
    
    # Drop columns with a proportion of null values greater than col_cutoff
    cols_to_drop = [col for col in df.columns if df[col].notnull().mean() < prop_required_column]
    df.drop(columns=cols_to_drop, inplace=True)

    # Drop rows with a proportion of null values greater than row_cutoff
    rows_to_drop = [row for row in df.index if df.loc[row].notnull().mean() < prop_required_row]
    df.drop(index=rows_to_drop, inplace=True)

    # Return the new DataFrame
    return df

def outlier_bound_calculator(df,k=1.5):
    '''
        This function calculates the lower and upper bound 
        to detect outliers and prints them for every numerical
        column.
    '''
    # list of only numerical columns
    continuous,_ = separate_column_type_list(df)
    
    # loop to print all bounds for each column
    for column in continuous:
        # calculating quartiles, IQR, and bounds
        quartile1, quartile3 = np.percentile(df[column], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = quartile1 - (k * IQR_value)
        upper_bound = quartile3 + (k * IQR_value)
        
        print(f'For {column} the lower bound is {lower_bound} and  upper bound is {upper_bound}')

def separate_column_type_list(df):
    '''
        Creates 2 lists separating continous & discrete
        variables.
        
        Parameters
        ----------
        df : Pandas DataFrame
            The DataFrame from which columns will be sorted.
        
        Returns
        ----------
        continuous_columns : list
            Columns in DataFrame with numerical values.
        discrete_columns : list
            Columns in DataFrame with categorical values.
    '''
    continuous_columns = []
    discrete_columns = []
    
    for column in df.columns:
        if (df[column].dtype == 'int' or df[column].dtype == 'float') and ('id' not in column) and (df[column].nunique()>10):
            continuous_columns.append(column)
        elif(df[column].dtype == 'int' or df[column].dtype == 'float') and (df[column].nunique()>11):
            continuous_columns.append(column)
        else:
            discrete_columns.append(column)
            
    return continuous_columns, discrete_columns

def get_distributions(df):
    '''
        Given numerical columns in a DataFrame:
        This function will plot distributions
        for all columns.
    '''
    cols,_ = separate_column_type_list(df)
    for col in cols:
        plt.hist(df[col])
        plt.title(f"distribution {col}")
        plt.show()
        
def summary(df):
    '''
        All-In-One Explore Phase Function
    '''
    # saving output of .isnull().sum() to a DataFrame
    null_count = df.isnull().sum().reset_index(name='count')
    
    # setting a name to columns in new DataFrame
    null_count.columns = ['column_name', 'num_rows_missing']

    # total count of nulls in DataFrame
    total_count = df.shape[0]
    
    # creating column = 'pct_row_missing' = vector op with calculated percent value 
    # using null_count and total_count
    null_count['pct_rows_missing'] = (null_count['num_rows_missing']/total_count)
    
    #return null_count[null_count['num_rows_missing']!=0]
    return null_count

def upper_outlier_detector(dataframe,column,k=1.5):
    '''
        Accepts numerical values from dataframe column
        Will return a series holding value for distance 
        from lower bound, but for a specific column
    '''
    q1,q3 = dataframe[column].quantile([0.25,0.75])
    iqr = q3-q1
    upper_bound = q3+k*iqr
    
    return np.where(dataframe[column]>upper_bound,1,0)

def lower_outlier_detector(dataframe,column,k=1.5):
    '''        
        Accepts numerical values from dataframe column
        Will return a series holding value for distance 
        from lower bound, but for a specific column
    '''
    q1,q3 = dataframe[column].quantile([0.25,0.75])
    iqr = q3-q1
    
    lower_bound=q1-k*iqr

    return np.where(dataframe[column]<lower_bound,1,0)

def lower_outliers(df):
    '''
        uses lower_outlier_detector() and selects only
        numerical columns from incoming DataFrame
    '''
    num, _ = separate_column_type_list(df)
    
    outliers = pd.DataFrame()
    
    for col in num:
    # creates a new column with suffix `_lower_outliers` and adds 1 or 0
        outliers[f'{col}_lower_outliers'] = lower_outlier_detector(df,col)

    return outliers.sum().sort_values(ascending=False)
def upper_outliers(df):
    '''
        uses upper_outlier_detector() and selects only
        numerical columns from incoming DataFrame
    '''
    num, _ = separate_column_type_list(df)
    
    outliers = pd.DataFrame()
    
    for col in num:
    # creates a new column with suffix `_upper_outliers` and adds 1 or 0
        outliers[f'{col}_upper_outliers'] = upper_outlier_detector(df,col)

    return outliers.sum().sort_values(ascending=False)
def all_outliers(df):
    '''
        Given numerical columns in a DataFrame:
        This function will give you total count of 
        all outliers outside both Uppler & Lower bound.
        Sorted Greatest to Least.
    '''
    # calculate interquartile range
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # detect outliers
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    return outliers.sum().sort_values(ascending=False)

def all_outliers_split_bounds(df):
    '''
        Given a DataFrame this function will
        select all numerical columns.
        Appends all outlier to new column in
        DataFrame & outputs a summary
    '''
    num, _ = separate_column_type_list(df)
    summary = pd.DataFrame()
    
    for col in num:
    # creates a new column with suffix `_upper_outliers` and adds 1 or 0
    # using lower or upper_outlier_detector()
        summary[f'{col}_upper_outliers'] = upper_outlier_detector(df,col)
        summary[f'{col}_lower_outliers'] = lower_outlier_detector(df,col)

    return summary.sum().sort_values(ascending=False)

def rename_columns(df):
    # rename all columns to `snake_case`
    # lower() all columns, then replace() spaces with underscores
    return df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

def data_dictionary(df):
    # Printing a data dictionary using a printout of each column name
    # formatted as a MarkDown table
    # =================================================================

    # variable to hold size of longest string in dataframe column names
    size_longest_name = len((max((df.columns.to_list()), key = len)))

    # head of markdown table
    print(f"| {'Name' : <{size_longest_name}} | Definition |")
    print(f"| {'-'*size_longest_name} | {'-'*len('Definition')} |")

    # dataframe column content
    for i in (df.columns.to_list()):
        print(f"| {i : <{size_longest_name}} | Definition |")

def handle_outliers(df,col_list,k):
    ''' 
        This function takes in a dataframe, the threshold and a list of columns 
        and returns the dataframe with outliers removed
    '''   
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def get_boxplot_distributions(df):
    num,_ = separate_column_type_list(df)
    for col in num:
    
        plt.boxplot(df[col])
        plt.title(f'distribution of {col}')
        plt.show()
    
def question_hypothesis(question_number,df,column_name,target,alpha=.05):
    df[column_name]
    
    display(Markdown(f"# Question #{question_number}:"))
    display(Markdown(f"### Hypothesis:"))
    display(Markdown(f"$H_0$: There is no correlation with `{column_name}` to `{target}`"))
    display(Markdown(f"$H_A$: There is a correlation between `{column_name}` and `{target}` "))
    #r, p = stats.pearsonr(df[column_name], df[target])
    stats.ttest
    display(Markdown(f"### Statistics Test:"))
    display(Markdown(f"### `Pearson's R = {r}`"))

    display(Markdown(eval_results(p, alpha, column_name, target)))
    
def get_dummies(df,column):
    to_dummy=column
    dummies = pd.get_dummies(df[to_dummy], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df