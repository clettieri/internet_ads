import pandas as pd
import numpy as np

def get_column_names(file_name, first_col_is_label=True):
    '''(string) -> list of strings
    
    Will open a text file assumed to contain 1 column name
    per line.  Will remove all blank lines.  And will strip
    rest of line after a ':' is reached. Returns a list of
    column names.
    '''
    col_names = []
    #Since data file uses carriage return, open 'rU'
    with open(file_name, 'rU') as f:
        for line in f:
            #If not blank line
            if line.strip():
                #Get column name until ':'
                col_names.append(line.strip().split(':')[0])
    if first_col_is_label:
        #Col List has label as first column
        col_names.pop(0)
        #DataFrame has label as last column
        col_names.append('is_ad')
    return col_names

def convert_label_to_numeric(df):
    '''(DataFrame) -> DataFrame
    
    Will convert the label column 'Is_ad' into numeric
    values. 1 representing an ad, 0 representing non-ad.
    '''
    df['is_ad'] = np.where(df['is_ad']=='ad.', 1, 0)  
    return df

def load_data_with_col_names(data_file, column_file):
    '''(string, string) -> DataFrame
    
    Will load column names from a text file and then read the
    data file as csv.  Return a dataframe of data file
    with proper column names.
    '''
    col_names = get_column_names(column_file)
    #Missing values in the data file are a '?' with varying whitespace
    df = pd.read_csv(data_file, names=col_names, skipinitialspace=True, na_values=['?'])
    df = convert_label_to_numeric(df)
    return df