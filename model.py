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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from IPython.display import display, Markdown


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



def get_dum1(df):   
    # Create dummies for non-binart categorical variables
    to_dummy = ['wine_color']
    dummies = pd.get_dummies(df[to_dummy], drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    
    # drop redundant column
    drop = ['wine_color']
    df.drop(columns=drop, inplace=True)
    
    return df
    
def get_dum2(df1,df2,df3): 
    to_dummy = ['wine_color']
    # Train
    train_dummies = pd.get_dummies(df1[to_dummy], drop_first=False)
    df1 = pd.concat([df1, train_dummies], axis=1)

    # Validate
    validate_dummies = pd.get_dummies(df2[to_dummy], drop_first=False)
    df2 = pd.concat([df2, validate_dummies], axis=1)

    # Test
    test_dummies = pd.get_dummies(df3[to_dummy], drop_first=False)
    df3 = pd.concat([df3, test_dummies], axis=1)
    # Drop Old
    drop = ['wine_color']
    df1.drop(columns=drop, inplace=True)
    df2.drop(columns=drop, inplace=True)
    df3.drop(columns=drop, inplace=True)
    
    return df1,df2,df3


def get_baseline(df):
    df['baseline'] = df['quality'].value_counts().idxmax()
    (df['quality'] == df['baseline']).mean()
    # clean f string markdown
    display(Markdown(f"### Baseline: {(df['quality'] == df['baseline']).mean()*100:.2f}%"))

def get_rf(X_train, y_train, X_validate, y_validate):   
    rf2 = RandomForestClassifier(max_depth=3, random_state=42,
                            max_samples=0.5)
    #fit it 
    rf2.fit(X_train, y_train)
    # clean f string
    print('Random Forest Model')
    print(f"Accuracy of Random Forest on train data: {rf2.score(X_train, y_train)}") 
    print(f"Accuracy of Random Forest on validate: {rf2.score(X_validate, y_validate)}")
    
    
def get_logit(X_train, y_train, X_validate, y_validate):
    logit2 = LogisticRegression(C=.1, random_state=42, 
                           intercept_scaling=1, solver='newton-cg')

    #fit the model
    logit2.fit(X_train, y_train, )
    #clean f string
    print('Logistic Regression Model')
    print(f"Accuracy of Logistic Regression on train: {logit2.score(X_train, y_train)}") 
    print(f"Accuracy of Logistic Regression on validate: {logit2.score(X_validate, y_validate)}")


def get_logit2(X_train, y_train, X_validate, y_validate):
    logit3 = LogisticRegression(C=100, random_state=42, 
                           intercept_scaling=1, solver='newton-cg')

    #fit the model
    logit3.fit(X_train, y_train)
    #clean f string
    print('Logistic Regression Model')
    print(f"Accuracy of Logistic Regression on train: {logit3.score(X_train, y_train)}") 
    print(f"Accuracy of Logistic Regression on validate: {logit3.score(X_validate, y_validate)}")
    
        

def get_knn(X_train, y_train, X_validate, y_validate):
    knn3 = KNeighborsClassifier(n_neighbors=21)
    knn3.fit(X_train, y_train)
    knn3.score(X_train, y_train)
    knn3.score(X_validate, y_validate)
    # clean f string
    print('KNN Model')
    print(f"Accuracy of KNN on train: {knn3.score(X_train, y_train)}") 
    print(f"Accuracy of KNN on validate: {knn3.score(X_validate, y_validate)}")
    
    

def get_test1(X_test, y_test):
    
    test_score = logit3.score(X_test, y_test)

    # clean f string
    print('Logistic Regression Model')
    print(f'Accuracy on Test {test_score:.2f}')
    

def get_test2(X_train, y_train, X_test, y_test):
    # Recalculating Best Peforming Model with new name
    best_model = LogisticRegression(C=100, random_state=42, 
                               intercept_scaling=1, solver="newton-cg")    
    best_model.fit(X_train, y_train)
    best_model.score(X_test,y_test)
    
     # clean f string
    display(Markdown(f'### Logistic Regression Model'))
    display(Markdown(f'### Accuracy on Test {best_model.score(X_test,y_test)*100:.2f}%'))
    

def get_mvb(X_train, y_train, X_test, y_test, wines):
    # Recalculating Best Peforming Model with new name
    best_model = LogisticRegression(C=100, random_state=42, 
                               intercept_scaling=1, solver="newton-cg")    
    best_model.fit(X_train, y_train)
    best_model.score(X_test,y_test)
    
    # Baseline
    plot_baseline = (wines['quality'] == wines['baseline']).mean() 
    
    # Best Performing Model(Logistic Regression Combo{c=100,newton-cg}) Test Score: 
    best_test_score = best_model.score(X_test,y_test)  
    
    # Test Scores: Project Baseline vs Best Model
    plot_baseline, best_test_score
    
    # Temporary Dictionary Holding Baseline & Model Test Score
    best_model_plot={"Baseline":[plot_baseline], "Test":[best_test_score]}
    
    # Converting Temporary Dictionary to DataFrame
    best_model_plot = pd.DataFrame(best_model_plot)
    
    # Visualizing Both Baseline & Model Test Scores
    
    fig=sns.barplot(data= best_model_plot,palette="colorblind")
    plt.title("Best Model vs. Baseline")
    fig.set(ylabel='Scores')
    plt.show()
    

def get_top_models(X_train, y_train, X_validate, y_validate):
    # Best Random Forest
    best_rf = RandomForestClassifier(max_depth=3, random_state=42, max_samples=0.5)
    best_rf.fit(X_train, y_train)

    best_rf_train_score = best_rf.score(X_train, y_train)
    best_rf_validate_score = best_rf.score(X_validate, y_validate)
    
    # Best KNN
    best_knn = KNeighborsClassifier(n_neighbors=21)
    best_knn.fit(X_train, y_train)

    best_knn_train_score = best_knn.score(X_train, y_train)
    best_knn_validate = best_knn.score(X_validate, y_validate)
    
    # Best Model: Logistic Regression
    best_lr1 = LogisticRegression(C=.1, random_state=42,intercept_scaling=1, solver='newton-cg')
    best_lr1.fit(X_train, y_train)

    best_lr1_train_score = best_lr1.score(X_train, y_train)
    best_lr1_validate_score = best_lr1.score(X_validate, y_validate)
    
    # Best Model: Combo - Logistic Regression
    best_lr2 = LogisticRegression(C=100, random_state=42,intercept_scaling=1, solver="newton-cg")    
    best_lr2.fit(X_train, y_train)

    best_lr2_train_score = best_lr2.score(X_train, y_train)
    best_lr2_validate_score = best_lr2.score(X_validate, y_validate)
    
    # Best Model: Combo - Logistic Regression
    best_lr2 = LogisticRegression(C=100, random_state=42,intercept_scaling=1, solver="newton-cg")    
    best_lr2.fit(X_train, y_train)

    best_lr2_train_score = best_lr2.score(X_train, y_train)
    best_lr2_validate_score = best_lr2.score(X_validate, y_validate)
    
    # lists with model names & score information
    best_model_name_list = ["KNN","Random_Forest","Logistic_Regression","Logistic_Regression_Combo"]
    best_model_train_scores_list = [best_knn_train_score,best_rf_train_score,best_lr1_train_score,best_lr2_train_score]
    best_model_validate_scores_list = [best_knn_validate,best_rf_validate_score,best_lr1_validate_score,best_lr2_validate_score]
    
    # new empty DataFrame
    best_scores_df = pd.DataFrame()
    
    # new columns using lists for data
    best_scores_df["Model"] = best_model_name_list
    best_scores_df["Train_Score"] = best_model_train_scores_list
    best_scores_df["Validate_Score"] = best_model_validate_scores_list
    
    melted_scores_df = pd.read_csv('melted_scores_df.csv')  
    
    train_max = melted_scores_df[melted_scores_df["Type"]=="Train"]["Score"].max()
    validate_max = melted_scores_df[melted_scores_df["Type"]=="Validate"]["Score"].max()
    
    plt.figure(figsize=(11, 8.5))

    # barplot
    top_model_scores_barplot = sns.barplot(data=melted_scores_df,x="Model",y="Score",hue="Type",palette="husl")

    # extras
    # ======
    # horizontal line
    plt.axhline(y = validate_max, color = 'purple', linestyle = '--')

    # formatting horizontal line text
    top_model_scores_barplot.text(2.63,.51, "53.2%",fontdict=dict(fontsize=15,color="white",weight="bold"))

    # moving legend
    sns.move_legend(top_model_scores_barplot, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title="Test Type", frameon=False)

    # formatting legend
    plt.setp(top_model_scores_barplot.get_legend().get_texts(), fontsize='15')
    plt.setp(top_model_scores_barplot.get_legend().get_title(), fontsize='20')
    #plt.legend(fontsize='x-large', title_fontsize='40')
    plt.show()


def plot_clust(df):
    num, cat = separate_column_type_list(df)
    train_scaled = df[num]
    # Create Object
    mm_scaler = MinMaxScaler()
    train_scaled[num] = mm_scaler.fit_transform(train_scaled[num])
    seed = 42
    cluster_count = 4

    kmeans = KMeans(n_clusters=cluster_count,random_state=seed)
    kmeans.fit(train_scaled)
    df['clusters']=kmeans.predict(train_scaled)
    sns.boxplot(data=df,x='clusters',y='alcohol',hue='quality')
    plt.title("What about Clustering?")
    plt.show()
    

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


def eval_results_2(p, alpha, group1, group2):
    '''
        Test Hypothesis  using Statistics Test Output.
        This function will take in the p-value, alpha, and a name for the 2 variables
        you are comparing (group1 and group2) and return a string stating 
        whether or not there exists a relationship between the 2 groups. 
    '''
    if p < alpha:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Reject $H_0$"))
        display(Markdown( f'There exists some relationship between {group1} and {group2}. (p-value: {p:.4f})'))
    
    else:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Failed to Reject $H_0$"))
        display(Markdown( f'There is not a significant relationship between {group1} and {group2}. (p-value: {p:.4f})'))
        


def cluster_hypothesis_test(question_number,df,question,column_name,target,alpha=.05):
        # calculations
        observed = pd.crosstab(df[column_name], df[target])
        chi2, p, degf, expected = stats.chi2_contingency(observed)
        value = chi2
        p_value = p
        
        # Output variables
        test = "Chi-Square"

        # Markdown Formatting
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"# {question}"))
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no relationship between `{column_name}` to `{target}`"))
        display(Markdown(f"$H_A$: There is a relationship between `{column_name}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))

        eval_results_2(p_value, alpha, column_name, target)
