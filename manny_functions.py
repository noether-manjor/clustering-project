def question_hypothesis_test(question_number,df,column_name,target,alpha=.05):
    num, cat = separate_column_type_list(df)
    
    if (target in cat) and (column_name in num):
        # calculation
        overall_alcohol_mean = wines[column_name].mean()
        quality_sample = wines[wines[target] >= 7][target]
        t, p = stats.ttest_1samp(quality_sample, overall_alcohol_mean)
        value = t
        p_value = p/2
        
        # Output variables
        test = "1-Sample T-Test"

        # Markdown Formatting
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no correlation with `{column_name}` to `{target}`"))
        display(Markdown(f"$H_A$: There is a correlation between `{column_name}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))

        display(Markdown(eval_results(p_value, alpha, column_name, target)))
        
    elif (target in cat) and (column_name in cat):
        # calculations
        observed = pd.crosstab(df[column_name], df[target])
        chi2, p, degf, expected = stats.chi2_contingency(observed)
        value = chi2
        p_value = p
        
        # Output variables
        test = "Chi-Square"

        # Markdown Formatting
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no correlation with `{column_name}` to `{target}`"))
        display(Markdown(f"$H_A$: There is a correlation between `{column_name}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))

        display(Markdown(eval_results(p_value, alpha, column_name, target)))
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

