# HTS to RNAcompete Prediction

This project develops a method for predicting RNAcompete (RNAC) results for a given protein on a specified probe list, using the protein's HTR-SELEX (HTS) experiment results.

## Project Overview

RNA-binding proteins (RBPs) play crucial roles in post-transcriptional gene regulation. This project aims to bridge the gap between two experimental methods for studying RBP-RNA interactions: HTR-SELEX and RNAcompete. By leveraging machine learning, we predict RNAcompete results based on HTR-SELEX data, potentially reducing the need for additional experimental work.

## Key Results

Our model achieved promising results:

- Mean Squared Error (MSE) of 0.102 on both validation and training sets
- MSE of 0.113 on the test set
- Mean correlation of 0.692 between predicted and actual RNA intensities across the training set proteins

These results indicate a strong predictive capability of our model, demonstrating its potential in accurately forecasting RNAcompete results from HTR-SELEX data.

## Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <project_directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the prediction, use the following command:

```sh
python main.py <output_file> <rnacompete_file> <selex_files>

Replace `<output_file>`, `<rnacompete_file>`, and `<selex_files>` with your specific file paths.
