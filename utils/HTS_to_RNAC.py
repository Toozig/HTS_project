import os
import json
import numpy as np
import pandas as pd
from utils.HTS_data_processing import oneHot_encode, SEQ_LEN
from xgboost import XGBRegressor
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras import backend as K
from typing import List
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import concurrent.futures
from keras.layers import Layer
from keras.saving import register_keras_serializable
# path to predictions of RBP1-38 on the RNAC dataset

HTS_RNAC_PRED = 'data/first_xy_predictions'
RNAC_TO_RNAC_MODEL = 'data/final_models/rnac_to_rnac.keras'
MODEL_DIR = 'data/final_models'
HTS_TO_RNAC_SCALED_MODEL = 'HTS_to_RNAC_scaled.keras'
HTS_TO_RNAC_MODEL = 'HTS_to_RNAC.keras'
XGB_MODEL_SCALED = 'xgb_model_HTS_to_RNAC_scaled.json'
XGB_MODEL = 'xgb_model_HTS_to_RNAC.json'
RNAC_INTENSITIES_DIR = 'data/RNAcompete_intensities'

def load_xy_dict():
    if not os.path.exists(HTS_RNAC_PRED):
        raise FileNotFoundError(f"The directory {HTS_RNAC_PRED} does not exist.")
    
    xy_dict = {}
    for filename in os.listdir(HTS_RNAC_PRED):
        if filename.endswith('.json'):
            xy_dict_path = os.path.join(HTS_RNAC_PRED, filename)
            with open(xy_dict_path, 'r') as f:
                dict_loaded = json.load(f)
                key = dict_loaded['id']
                cur_X = dict_loaded['X']
                cur_y = dict_loaded['y']
                xy_dict[key] = (np.array(cur_X), np.array(cur_y))

    return xy_dict


def get_rnac_data(RNAC_sequence_file: str) -> pd.DataFrame:
    with open(RNAC_sequence_file, 'r') as f:
        RNAC_sequences = f.readlines()
    RNAC_sequences = [seq.strip() for seq in RNAC_sequences] 
    RNAC_sequences

    rnac_df = pd.DataFrame()
    rnac_result_path = RNAC_INTENSITIES_DIR
    for i in os.listdir(rnac_result_path):
        if not i.endswith('.txt'):
            continue
        rnac_result_file = os.path.join(rnac_result_path, i)
        rnac_result = pd.read_csv(rnac_result_file, header=None)
        rnac_result.columns = [i.split('.')[0] ]
        rnac_df[i.split('.')[0]] = rnac_result.iloc[:]
        
    rnac_df.index = RNAC_sequences
    rnac_df  =rnac_df.sort_index(axis=1)
    return rnac_df
  

def get_hts_onehot(RNAC_sequences):
    RNAC_sequences = [seq.ljust(SEQ_LEN,'N')for seq in RNAC_sequences]
    hts_result = [oneHot_encode(seq) for seq in RNAC_sequences]
    hts_result = np.array(hts_result)
    return hts_result


def get_X(rnac_df, model,test_score):
    hts_result = get_hts_onehot(rnac_df.index.values)
    cur_pred = model.predict(hts_result)
    cur_X = cur_pred * rnac_df
   # if RPB n is less than 39 - save the prediction
    cur_X = cur_X.values
    cur_X = np.concatenate((np.full((cur_X.shape[0], 1), test_score, cur_X)), axis=1)
    return cur_X


def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_xgboost_model(model_path):
    model = XGBRegressor()
    model.load_model(model_path)
    return model


@register_keras_serializable()
class SaveFirstNumber(Layer):
    def __init__(self, **kwargs):
        super(SaveFirstNumber, self).__init__(**kwargs)
        self.saved_number = None

    def call(self, inputs):
        self.saved_number = inputs[:, 0:1]
        return inputs

    def get_config(self):
        config = super(SaveFirstNumber, self).get_config()
        return config

import keras

class MultiplyBySavedNumber(Layer):
    def __init__(self, saved_number_layer, **kwargs):
        super(MultiplyBySavedNumber, self).__init__(**kwargs)
        self.saved_number_layer = saved_number_layer

    def call(self, inputs):
        return inputs * self.saved_number_layer.saved_number

    def get_config(self):
        config = super(MultiplyBySavedNumber, self).get_config()
        config.update({
            'saved_number_layer': self.saved_number_layer.name  # Save the name of the layer
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Retrieve the layer by name from the model's layers
        saved_number_layer = keras.layers.deserialize(config.pop('saved_number_layer'))
        return cls(saved_number_layer, **config)

def load_all_models():

    NN_scaled = tf.keras.models.load_model(f'{MODEL_DIR}/{HTS_TO_RNAC_SCALED_MODEL}',
                                            {'SaveFirstNumber': SaveFirstNumber, 'MultiplyBySavedNumber': MultiplyBySavedNumber})
    NN = tf.keras.models.load_model(f'{MODEL_DIR}/{HTS_TO_RNAC_MODEL}',
                                     {'SaveFirstNumber': SaveFirstNumber, 'MultiplyBySavedNumber': MultiplyBySavedNumber})
    xgb_scaled = load_xgboost_model(f'{MODEL_DIR}/{XGB_MODEL_SCALED}')
    xgb = load_xgboost_model(f'{MODEL_DIR}/{XGB_MODEL}')
    return NN_scaled, NN, xgb_scaled, xgb


def predict_model_helper(model, data):
    return model.predict(data).flatten()

def HTS_to_RNAC_score(HTS_X):
    NN_scaled, NN, xgb_scaled, xgb = load_all_models()

    models = {
        'NN_scaled_score': NN_scaled,
        'NN_score': NN,
        'xgb_scaled_score': xgb_scaled,
        'xgb_score': xgb
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {name: executor.submit(predict_model_helper, model, HTS_X) for name, model in models.items()}
        results = {name: future.result() for name, future in futures.items()}
    
    order = ['NN_score', 'xgb_score', 'NN_scaled_score', 'xgb_scaled_score']
    result = pd.DataFrame(results)
    return result[order].values


def pearson_correlation_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = r_num / r_den
    return 1 - r  # We want to minimize the negative correlation


def predict_RNAC(RNAC_X: np.ndarray) -> List[float]:
    final_model = load_model(RNAC_TO_RNAC_MODEL, 
                             custom_objects={'pearson_correlation_loss': pearson_correlation_loss})
    RNAC_Y_pred = final_model.predict(RNAC_X)
    return RNAC_Y_pred.flatten().astype(str).tolist()


def save_HTS_X_pred(HTS_X:np.ndarray,protein_name:str):
    prediction_dir = 'data/final/predictions'
    cur_data = { protein_name : HTS_X.tolist()}
    os.makedirs(prediction_dir, exist_ok=True)
    with open(f'{prediction_dir}/{protein_name}.json', 'w') as f:
        json.dump(cur_data, f)

def get_prediction(hts_model: Sequential,
                                 prtotein_name: str,
                                 test_score: float,
                                 RNAC_seq_path: str) -> List[float]:
    RNAC_df = get_rnac_data(RNAC_seq_path)
    HTS_X = get_X(RNAC_df, prtotein_name, hts_model, test_score)
 
    RNAC_X = HTS_to_RNAC_score(HTS_X)
    RNAC_Y_pred =  predict_RNAC(RNAC_X[np.newaxis, :])
    return RNAC_Y_pred
