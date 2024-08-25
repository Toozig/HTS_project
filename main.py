import sys
import argparse
from typing import List, Tuple, Dict
from pandas import DataFrame
from utils.HTS_data_processing import process_HTS_raw_files, impute_missing_cycles
from utils.HTS_model import train_hts_model
from utils.HTS_to_RNAC import  get_hts_onehot
from utils.RNAC_models import get_full_HTS_to_RNAC_prediction, get_big_model_HTS_to_RNAC_prediction
import numpy as np
from tensorflow.keras.models import Model
import pandas as pd
import os

DEBUG = True


def write_scores(output_file:str, scores:np.ndarray) -> None:
    """
    Writes the scores to an output file.

    Args:
        output_file (str): Path to the output file.
        scores (List[float]): List of scores to write.
    """
    with open(output_file, 'w') as file:
        file.write("\n".join(scores.flatten().astype(str)))



def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process SELEX and RNCMPT files.")
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("rnacompete_file", type=str, help="The RNCMPT file path.")
    parser.add_argument("selex_files", type=str, nargs='+', help="The SELEX file paths (1 to 5 files).")
    
    args = parser.parse_args()

    # Check the number of SELEX files
    if len(args.selex_files) < 1:
        parser.error("You must provide between 1 and 5 SELEX files.")
    
    return args


from utils.HTS_to_RNAC import get_rnac_data, get_X

def get_X_to_save(protein_name,rnac_df, model,test_score, to_save = False):
    hts_result = get_hts_onehot(rnac_df.index.values)
    cur_pred = model.predict(hts_result)
    cur_X = cur_pred * rnac_df
    if protein_name in cur_X.columns: #  mix the data if it is in the train set
        cur_X[protein_name] = cur_X[protein_name].sample(frac=1).values
        save_dir = 'data/final/train_predictions'
    else:
        save_dir = 'data/final/test_predictions'
    os.makedirs(save_dir, exist_ok=True)
    test_score_column = np.full((cur_X.shape[0], 1), test_score)
    cur_X.insert(0, 'test_score', test_score_column)

    cur_X.to_csv(f'{save_dir}/{protein_name}.csv')  
    return cur_X



def get_HTS_RNAC_probe_prediction(protein_name: str,rnac_df: DataFrame, model:Model,test_score: float) -> DataFrame:
    hts_result = get_hts_onehot(rnac_df.index.values)
    cur_pred = model.predict(hts_result)
    cur_X = cur_pred * rnac_df
    if protein_name in cur_X.columns: #  mix the data if it is in the train set
        cur_X[protein_name] = cur_X[protein_name].sample(frac=1).values
    test_score_column = np.full((cur_X.shape[0], 1), test_score)
    cur_X.insert(0, 'test_score', test_score_column)  
    return cur_X



def prepare_hts_data(protein_name:str, selex_files: List[str]) -> Tuple[DataFrame, bool]:
    """
    Prepares the HT-SELEX data for training.

    Args:
        protein_name (str): The name of the protein.
        selex_files (List[str]): List of paths to the SELEX files.

    Returns:
        Tuple[DataFrame, bool]: A tuple containing the HT-SELEX data and a boolean indicating if imputation is needed.
    """
    # Step 2: Process the SELEX files into the model input format
    hts_df = process_HTS_raw_files(selex_files)

    # Step 2.5: Impute missing cycles if necessary
    need_imputation = len(selex_files) == 1
    if need_imputation:
        print("Imputing missing cycles...")
        hts_df = impute_missing_cycles(protein_name,hts_df)

    return hts_df, need_imputation

def run_prediction(output_file: str, rnacompete_file: str, selex_files: List[str]) -> None:
    
    # Extract the protein name from the first SELEX file
    protein_name = selex_files[0].split('/')[-1].split('_')[0]

    # print input if DEBUG
    if DEBUG:
        print(f"Protein name: {protein_name}")
        print(f"Output file: {output_file}")
        print(f"RNAcompete file: {rnacompete_file}")
        print(f"SELEX files: {selex_files}")


    # Step 2: Process the SELEX files into the model input format
    hts_df, is_imputed = prepare_hts_data(protein_name, selex_files)

    # Step 3: train the model
    model, history, test_loss = train_hts_model(hts_df, is_imputed)
    RNAC_df = get_rnac_data(rnacompete_file)
    HTS_X = get_HTS_RNAC_probe_prediction(protein_name, RNAC_df,  model, test_loss)
    print(f"Saved the prediction for {protein_name}")
    # RNAC_pred = get_full_HTS_to_RNAC_prediction(HTS_X) # probelm to load the last  model
    RNAC_pred = get_big_model_HTS_to_RNAC_prediction(HTS_X)
    write_scores(output_file, RNAC_pred)
    print(f'done prediction on {protein_name}') 
    print(f"Scores written to {output_file}.")  


def main():
    args = parse_arguments()

    output_file = args.output_file
    rnacompete_file = args.rnacompete_file
    selex_files = args.selex_files
    run_prediction(output_file, rnacompete_file, selex_files)
    print


if __name__ == "__main__":
    main()
