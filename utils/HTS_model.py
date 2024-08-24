import numpy as np
from typing import List, Tuple
import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
from .HTS_data_processing import process_HTS_df
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, Callback
import json
import os
# import keras history object
from keras.callbacks import History

DEBUG = True


# Data processing variables
MAX_N =4
SEQ_LEN = 42

# Model variables
STRIDES = 1
RELU = 'relu'
LINEAR = 'linear'
SIGMOID = 'sigmoid'
MSE = 'mse'
MOTIF_LEN = 6
N_MOTIF = 124
L_RATE = 0.0066






# train variables
EPOCHS = 24
# original is 256 but it is too slow (RBP1 got Test Loss: 0.07756771147251129,
#                       Test Metric (Pearson Correlation): 0.6312665343284607) 0.07849756628274918, Test Metric (Pearson Correlation): 0.6251803040504456
BATCH_SIZE = 256
# Test Loss: 0.07811325043439865,
#  Test Metric (Pearson Correlation): 0.62741619348526
# BATCH_SIZE = 512



@register_keras_serializable()
def pearson_correlation(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Pearson correlation coefficient between the true and predicted values.

    Parameters:
    y_true (tf.Tensor): The ground truth values.
    y_pred (tf.Tensor): The predicted values.

    Returns:
    tf.Tensor: The Pearson correlation coefficient.
    """
    # Reshape y_true to ensure it is a column vector
    y_true = tf.reshape(y_true, (-1, 1))
    
    # Compute the Pearson correlation coefficient
    correlation = tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=-1)
    
    return correlation




def build_model(expirement_id: str, input_shape: Tuple[int, ...], compile: bool = True) -> Sequential:
    """
    Builds and compiles a Sequential model for the given experiment.

    Parameters:
    expirement_id (str): The identifier for the experiment.
    input_shape (Tuple[int, ...]): The shape of the input data.
    compile (bool): Whether to compile the model. Default is True.

    Returns:
    Sequential: The constructed Keras Sequential model.

    Model Architecture:
    - Input layer with the specified input shape.
    - Conv1D layer with N_MOTIF filters, kernel size of MOTIF_LEN, strides of STRIDES,
         and ReLU activation.
    - MaxPooling1D layer with pool size of (MOTIF_LEN - 1).
    - Flatten layer to convert the 2D output to 1D.
    - Dense layer with 32 units and ReLU activation.
    - Dense layer with 1 unit and linear activation.
    """
    model = Sequential(name=expirement_id)
    
    # Input layer
    model.add(Input(shape=input_shape))
    
    # Conv1D layer
    model.add(Conv1D(filters=N_MOTIF,
                     kernel_size=(MOTIF_LEN,),
                     strides=STRIDES,
                     activation=RELU))
    
    # MaxPooling1D layer
    model.add(MaxPooling1D(pool_size=(MOTIF_LEN - 1,)))
    
    # Flatten layer
    model.add(Flatten()) 
    
    # Dense layer with 32 units
    model.add(Dense(units=32, activation=RELU))
    
    # Dense layer with 1 unit
    model.add(Dense(units=1, activation=LINEAR)) 
    
    # Optimizer
    optimizer = Adam(learning_rate=L_RATE)
    
    # Compile the model if required
    if compile:
        model.compile(optimizer=optimizer, loss=MSE, metrics=[pearson_correlation])
    
    return model


def save_model_eval(eval_dict: dict, model_name: str):
    """
    Saves the model evaluation metrics to a file.
    """

    save_path = 'data/final/model_eval'
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/{model_name}_eval.json', 'w') as f:
        json.dump(eval_dict, f)



class NaNStopping(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_pearson = logs.get('val_pearson_correlation')
        if val_pearson is not None and np.isnan(val_pearson):
            print(f"Epoch {epoch + 1}: val_pearson_correlation is NaN. Stopping training.")
            self.model.stop_training = True

def train_model(model_name:str,
                X: np.ndarray,
                y: np.ndarray,
                weighted: bool,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS) -> Tuple[Sequential, History, dict]:
    """
    Trains the given model on the provided data.

    Parameters:
    X (np.ndarray): The input data.
    y (np.ndarray): The target data.
    weighted (bool): Whether to use weighted loss.
    batch_size (int): The batch size for training. Default is 256.
    epochs (int): The number of epochs to train for. Default is 100.

    Returns:
    Tuple[Sequential, : The trained model
    History,  training history
    dict]: ,  evaluation metrics.
    
    """
    max_retries = 5
    for attempt in range(max_retries):
        model = build_model(model_name, X.shape[1:])
        # Split the data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        # if the data is imputed, assign a higher weight to the positive class 
        # which is from the original data
        if weighted:
            sample_weights = np.where(y_train == 1, 1.5, 1.0)
        else:
            sample_weights = None

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        nan_stopping = NaNStopping()
        
        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, nan_stopping],
                            sample_weight=sample_weights)
        
        # Evaluate the model on the test set
        test_loss, test_pearson = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}, Test Metric (Pearson Correlation): {test_pearson}")
        eval_dict = {'test_loss': test_loss, 'test_pearson': test_pearson}
        
        # get the validation loss and pearson correlation
        val_loss = history.history['val_loss']
        val_pearson = history.history.get('val_pearson_correlation', [np.nan])
        
        # Check for NaN in val_pearson_correlation
        if not any(np.isnan(val_pearson)):
            eval_dict['val_loss'] = val_loss
            eval_dict['val_pearson'] = val_pearson
            eval_dict['train_loss'] = history.history['loss']
            eval_dict['train_pearson'] = history.history.get('pearson_correlation', [np.nan])
            if DEBUG:
                save_model_eval(eval_dict, model.name)
            return model, history, test_pearson
        else:
            print(f"Attempt {attempt + 1}/{max_retries} failed due to NaN in val_pearson_correlation. Retrying...")

    # save the protein_name
    save_file = 'data/final/failed_proteins.txt'
    # check if the file exists if not create it
    if not os.path.exists(save_file):
        with open(save_file, 'w') as f:
            f.write("protein_name\n")
    raise ValueError("Training failed after 5 attempts due to NaN in val_pearson_correlation.")

def train_hts_model(hts_df: pd.DataFrame, 
                    is_imputed: bool) ->  Tuple[Sequential, History, float]:
    """
    Trains a model on the given HTS data.

    Parameters:
    hts_df (pd.DataFrame): The HTS data.
    is_imputed (bool): Whether the data has been imputed.

    Returns:
    Tuple[Sequential, History, dict]: The trained model, training history,
                                         and evaluation metrics.
    """
    model_name = hts_df['protein'].values[0]
    X, y = process_HTS_df(hts_df)

    model, history, test_loss = train_model( model_name,X, y, weighted=is_imputed)
    return model, history, test_loss


