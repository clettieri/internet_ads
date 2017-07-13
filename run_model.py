import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

def load_data(file_name, label_col_index=-1):
    '''(string, int) -> array, array
    
    Given a file path and name to a csv file in the SAME FORMAT
    as the test data set, this will laod the file and return two
    arrays of values.  Label_col_index is the numerical index of the
    label column, default is -1 (last column in data file).
    '''
    df = pd.read_csv(file_name, skipinitialspace=True, na_values=['?'])
    #Get Features
    x_df = df.drop(df.columns[label_col_index], axis=1)
    X = x_df.values
    
    #Impute Missing x values
    imp = Imputer(strategy="median", axis=0)
    X = imp.fit_transform(X)
    
    #Get Labels
    df.iloc[:,label_col_index] = np.where(df.iloc[:,label_col_index]=='ad.', 1, 0)  
    y_df = df.iloc[:,label_col_index]
    y = y_df.values
    
    return X, y

def load_model(file_name):
    '''(string) -> sklearn classifier
    
    Given a file path and name to a joblib file, will return
    the sklearn classifer as python object.
    '''
    clf = joblib.load(file_name)
    return clf

def run_model(data_file, model_file, label_col_index=-1):
    '''(string, string, int) -> None
    
    Given a path to a data file in the same form as test data set,
    and to a joblib file containing the final model, this function will
    load the data and then run the model and score it.
    
    This function assumes that the labels will be included with the data
    file.
    '''
    #Load data
    X, y = load_data(data_file, label_col_index=label_col_index)
    #Load model
    try:
        clf = load_model(model_file)
    except:
        print "Error loading model file"
        print "Did you unzip 'final_model.zip'?"
    #Make predictions
    predictions = clf.predict(X)
    #Calculate score
    score = accuracy_score(predictions, y)
    print "Model Accuracy: " + str(score) 
    
if __name__ == "__main__":
    run_model("data.csv", "final_model.pkl")
    