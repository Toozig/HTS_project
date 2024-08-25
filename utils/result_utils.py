import numpy as np
from scipy.stats import pearsonr

def calculate_pearson_correlation(file1_path:str, file2_path:str)-> None:
    """
    Calculate the Pearson correlation between two files containing numbers.

    Args:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.  

    Returns:
        None
    """
    # Read numbers from the first file
    with open(file1_path, 'r') as file1:
        list1 = [float(line.strip()) for line in file1.readlines()]
    
    # Read numbers from the second file
    with open(file2_path, 'r') as file2:
        list2 = [float(line.strip()) for line in file2.readlines()]
    
    # Validate that both lists have the same length
    if len(list1) != len(list2):
        print("Error: The files do not contain the same number of lines.")
        return
    
    # Calculate Pearson correlation
    correlation, _ = pearsonr(list1, list2)
    
    # Print the Pearson correlation
    print(f"Pearson correlation: {correlation}")
