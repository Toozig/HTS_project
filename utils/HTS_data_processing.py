## HTS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List
import os
import requests
import numpy as np
import re

"""
The impuation dictionary is used to impute missing cycles for proteins with 1 cycle.
The dictionary is the output  of analysis which can be found in the notebook "comapre_enrichment/enrichment_analysis.ipynb"
"""
IMPUTATION_DICT = {
                'RBP54': 'RBP15',
                'RBP47': 'RBP38',
                'RBP55': 'RBP15',
                'RBP45': 'RBP13',
                'RBP17': 'RBP18',
                'RBP14': 'RBP13',
                'RBP5': 'RBP4'
                }

# path to  processed HTS data
MODEL_HTS_DATA = 'data/processed/HTS_csv_data/%s.csv'

# maximal value for appearance of a sequence in a cycle
MAX_N = 4

# maximal sequence length
SEQ_LEN = 42



def print_data_metadata(HTS_df: pd.DataFrame) -> None:

    """
    Prints metadata about the given DataFrame.

    Parameters:
    HTS_df (pd.DataFrame): DataFrame containing the sequences and their associated data.
    """
    max_cycle = HTS_df['cycle'].astype(int).max()
    min_cycle = HTS_df['cycle'].min()
    samples_per_cycle = HTS_df['cycle'].value_counts().to_dict()
    
    print(f"Max cycle: {max_cycle}")
    print(f"Min cycle: {min_cycle}")
    print(f"Samples per cycle: {samples_per_cycle}")

def process_HTS_raw_files(file_list: List[str]) -> pd.DataFrame:
    """
    Processes a list of HTS (High-Throughput Sequencing) files and combines them into a single DataFrame.

    Parameters:
    file_list (List[str]): List of file paths to the HTS files.

    Returns:
    pd.DataFrame: A DataFrame containing the combined data from all the HTS files.
    """
    # Extract the protein name from the first file's name
    protein_name = file_list[0].split('/')[-1].split('_')[0]
    
    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame()
    
    # Iterate over each file in the file list
    for file in file_list:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file, header=None)
        
        # Assign column names to the DataFrame
        df.columns = ['sequence', 'n']
        
        # Extract the cycle number from the file name and add it as a new column
        df['cycle'] = file.split('/')[-1].split('.')[0].split('_')[1]
        df['cycle'] = df['cycle'].astype(int)
        
        # Concatenate the current DataFrame with the result DataFrame
        result = pd.concat([result, df])
    
    # Add the protein name as a new column to the result DataFrame
    result['protein'] = protein_name
    print_data_metadata(result)
    return result

        


def download_and_load_rbp(protein_name):
    url = f"https://raw.githubusercontent.com/Toozig/HTS_to_RNAC/main/data/HTS_csv_data/{protein_name}.csv"
    local_path = f"/tmp/RBP{protein_name}.csv"
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the file to /tmp
        with open(local_path, 'wb') as file:
            file.write(response.content)
        
        # Load the file into a pandas DataFrame
        df = pd.read_csv(local_path, index_col=None)
        return df
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download the file: {e}")
    


def get_replacement_df(protein_name:str)-> pd.DataFrame:
    """
    Get the replacement DataFrame for the given protein name.
    Args:
        protein_name (str): Name of the protein (e.g. 'RBP54')
    """
    replacment_protein_name = IMPUTATION_DICT[protein_name]
    replacment_data = MODEL_HTS_DATA % replacment_protein_name
    if not os.path.exists(replacment_data):
        print(f"Downloading {replacment_data}")
        replacement_df = download_and_load_rbp(replacment_protein_name)
        return replacement_df
    replacement_df = pd.read_csv(replacment_data)
    return replacement_df

def impute_missing_cycles(protein_name:str, HTS_df: pd.DataFrame)-> pd.DataFrame:
    """
    Impute missing cycles for proteins with 1 cycle.
    Args:
        protein_name (str): Name of the protein (e.g. 'RBP54')
        HTS_df (pd.DataFrame) : DF contain

    """
    replacment_protein_name = IMPUTATION_DICT[protein_name]
    replacment_data = MODEL_HTS_DATA % replacment_protein_name
    replacement_df = pd.read_csv(replacment_data)
    # replace cycle 4 data with cycle 4 data from the model with high corrolation
    replacement_df = replacement_df[replacement_df.cycle != 4]
    new_data = pd.concat([HTS_df, replacement_df]).reset_index(drop=True)
    print_data_metadata(new_data)
    return new_data



def _filter_data(hts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the given DataFrame by removing outliers and duplicates.

    Parameters:
    hts_df (pd.DataFrame): DataFrame containing the sequences and their associated data.

    Returns:
    pd.DataFrame: A filtered DataFrame with outliers removed and duplicates handled.
    """
    # Drop outliers based on the 'n' column
    hts_df = hts_df[hts_df['n'] <= MAX_N]
    
    # If there are duplicates, keep the one with the highest 'cycle' value
    hts_df = hts_df.sort_values('cycle', ascending=False).drop_duplicates(subset='sequence', keep='first')
    
    return hts_df


def _is_valid_dna_sequence(sequence):
    pattern = re.compile('^[AGCTN]+$')
    if not pattern.match(str(sequence)):
        return False
    return True


def oneHot_encode(record:str) -> np.array:
    # RNA -> DNA (U -> T)
    string = record.upper().replace('\n', '').replace('U', 'T').replace(' ', '')
    if not _is_valid_dna_sequence(string):
        # replace invalid sequences with C #todo - check if this is the best way to handle invalid sequences
        string = re.sub('[^AGCTN]', 'N', string)
        print(f"Invalid DNA sequence - {record}! Replaced invalid characters with N")

    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4} 
    data = [mapping[char] for char in string.upper()]
    one_hot = np.eye(5)[data]

    # remove the last column (N)
    one_hot = one_hot[:, :4]
    return one_hot.astype(bool)


def oneHot_encode_data(hts_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    One-hot encodes the sequences in the given DataFrame and scales the target variable.

    Parameters:
    hts_df (pd.DataFrame): DataFrame containing the sequences and target variable 'cycle'.

    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing the one-hot encoded sequences (X) and the scaled target variable (y).
    """
    # Pad sequences to the same length
    hts_df['sequence'] = hts_df['sequence'].apply(lambda x: x.ljust(SEQ_LEN, 'N'))
    
    # One-hot encode the sequences
    X = np.array(list(map(oneHot_encode, hts_df['sequence'].tolist())))
    
    # Extract the 'cycle' column as the target variable
    y = hts_df['cycle'].values
    
    # Scale the target variable y into the range 0-1
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y


def process_HTS_df(hts_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Processes a list of HTS files and returns the one-hot encoded sequences and scaled target variable.

    Parameters:
    hts_df (pd.DataFrame): DataFrame containing the sequences and target variable

    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing the one-hot encoded sequences (X) and the scaled target variable (y).
    """
    # Filter the data by removing outliers and duplicates
    hts_df = _filter_data(hts_df)
    
    # One-hot encode the sequences and scale the target variable
    X, y = oneHot_encode_data(hts_df)
 
    return X, y