import datasets
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

## Loading the CoNaLa dataset
def load_data():
    # Load the dataset
    ds = load_dataset("neulab/conala", "curated")
    
    # Convert the train and test data to DataFrames
    train_data = pd.DataFrame(ds['train'])
    test_data = pd.DataFrame(ds['test'])
    
    # Combine train and test data
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    return data

def preprocess_data(data):
    # Drop duplicates and NaN values
    data.drop_duplicates(subset=['intent', 'snippet'], inplace=True)
    data.dropna(subset=['intent', 'snippet'], inplace=True)

    # Reset index
    data.reset_index(drop=True, inplace=True)

    # Train-validation-test split
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

data = load_data()
train_df, val_df, test_df = preprocess_data(data)

## Loading the MBPP data set 

import json
data1 = []
with open('/Users/vineeth/Desktop/Rutgers/Semester 3/NLP/Project/data.json', 'r') as file:
    for line in file:
        # Remove whitespace and parse each line as JSON
        data1.append(json.loads(line.strip()))

# Now `data` is a list of dictionaries where each item is a parsed JSON object from each line
print(data1)

# Converting to a dataframe 
df = pd.DataFrame(data1)

# Creating a new dataframe df1 with renamed columns
df1 = df[['task_id','text','test_setup_code','code']]

# Rename the columns in df1 to match data and then stack the matrix. (Ignore the index values)
df1.columns = ['question_id','intent','rewritten_intent','snippet']
df_stacked = pd.concat([data, df1], axis=0, ignore_index=True)

# Converting to csv files

# Unformatted MBPP dataset
df.to_csv("/Users/vineeth/Desktop/Rutgers/Semester 3/NLP/Project/mbpp.csv",index = False)

# MBPP formatted (Coloumn names changed, columns reduced)
df1.to_csv("/Users/vineeth/Desktop/Rutgers/Semester 3/NLP/Project/mbpp(formatted).csv", index = False)

# Final data set (mbpp + CoNaLa)
df_stacked.to_csv("/Users/vineeth/Desktop/Rutgers/Semester 3/NLP/Project/mbpp_conala.csv", index = False)