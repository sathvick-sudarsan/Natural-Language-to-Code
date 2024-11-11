import datasets
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_data():
    # Load the dataset
    ds = load_dataset("neulab/conala", "curated")
    
    # Convert the train and test data to DataFrames
    train_data = pd.DataFrame(ds['train'])
    test_data = pd.DataFrame(ds['test'])
    
    # Combine train and test data
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    return data

def load_data_from_csv(csv_file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.
    
    Args:
    - csv_file_path: str, path to the CSV file containing the data
    
    Returns:
    - data: pandas DataFrame containing the loaded data
    """
    # Load the dataset from the CSV file
    data = pd.read_csv(csv_file_path)
    
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
