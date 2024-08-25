import os
import numpy as np
import pandas as pd
from keras.models import Model
from utils.HTS_data_processing import oneHot_encode, SEQ_LEN
# path to predictions of RBP1-38 on the RNAC dataset

HTS_RNAC_PRED = 'data/first_xy_predictions'
MODEL_DIR = 'data/final_models'
HTS_TO_RNAC_SCALED_MODEL = 'HTS_to_RNAC_scaled.keras'
HTS_TO_RNAC_MODEL = 'HTS_to_RNAC.keras'
RNAC_INTENSITIES_DIR = 'data/RNAcompete_intensities'
RNAC_PARQUET_FILE = 'data/probe_intenseteis.parquet'
RNAC_DF_PARQUET_WEB_PATH = 'https://raw.githubusercontent.com/Toozig/HTS_to_RNAC/main/data/probe_intenseteis.parquet'



def load_RNAC_df()-> pd.DataFrame:
    """
    This function loads the RNAC dataset from a parquet file
    Returns:
    pd.DataFrame: The RNAC dataset


    """
    parquet_path = RNAC_PARQUET_FILE
    if not os.path.exists(parquet_path):
        print('Downloading RNAC data')
        parquet_path = '/tmp/probe_intenseteis.parquet'
        if not os.path.exists(parquet_path):
            # create the directory if it does not exist
            os.makedirs('/tmp', exist_ok=True)
            os.system(f'wget -O {parquet_path} {RNAC_DF_PARQUET_WEB_PATH}')
    rnac_df = pd.read_parquet(parquet_path, engine='pyarrow')    
    return rnac_df
        

def get_rnac_data(RNAC_sequence_file: str) -> pd.DataFrame:
    with open(RNAC_sequence_file, 'r') as f:
        RNAC_sequences = f.readlines()
    RNAC_sequences = [seq.strip() for seq in RNAC_sequences] 
    RNAC_sequences
    rnac_df = load_RNAC_df()
    rnac_df.index = RNAC_sequences
    return rnac_df
  
def get_hts_onehot(RNAC_sequences: str) -> np.ndarray:
    """
    This function takes a list of RNA sequences and returns the one-hot encoded sequences

    Parameters:
    RNAC_sequences (str): A list of RNA sequences

    Returns:
    np.ndarray: The one-hot encoded sequences

    """
    RNAC_sequences = [seq.ljust(SEQ_LEN,'N')for seq in RNAC_sequences]
    hts_result = [oneHot_encode(seq) for seq in RNAC_sequences]
    hts_result = np.array(hts_result)
    return hts_result

def get_X(rnac_df:pd.DataFrame, model: Model,test_score: float)-> np.ndarray:
    hts_result = get_hts_onehot(rnac_df.index.values)
    cur_pred = model.predict(hts_result)
    cur_X = cur_pred * rnac_df
   # if RPB n is less than 39 - save the prediction
    cur_X = cur_X.values
    cur_X = np.concatenate((np.full((cur_X.shape[0], 1), test_score, cur_X)), axis=1)
    return cur_X
