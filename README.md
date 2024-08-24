# HTS to RNAcompete Prediction

This project processes SELEX and RNAcompete files to predict RNA binding affinities using machine learning models.

## Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the prediction, use the following command:

```sh
python main.py <output_file> <rnacompete_file> <selex_files>