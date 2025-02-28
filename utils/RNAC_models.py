from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Add, Concatenate, Input, Layer
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import tensorflow as tf
import keras
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

HTS_TO_RNAC_COLS = ['small_model', 'xgboost_model', 'big_model']
COL_ORDER =['test_score', 'RBP1', 'RBP2', 'RBP3', 'RBP4', 'RBP5', 'RBP6', 'RBP7', 'RBP8', 'RBP9', 'RBP10', 'RBP11', 'RBP12', 'RBP13', 'RBP14', 'RBP15', 'RBP16', 'RBP17', 'RBP18', 'RBP19', 'RBP20', 'RBP21', 'RBP22', 'RBP23', 'RBP24', 'RBP25', 'RBP26', 'RBP27', 'RBP28', 'RBP29', 'RBP30', 'RBP31', 'RBP32', 'RBP33', 'RBP34', 'RBP35', 'RBP36', 'RBP37', 'RBP38']
COMPLEX_BIG_MODEL_PATH = 'data/models/hts_to_rnac_complex_modelV2.keras'
COMPLEX_BIG_MODEL_SCALER_PATH = 'data/models/HTS_to_RNAC_scaler.pkl'
BIG = 'big'
BIG_MODEL = 'big_model'


@keras.saving.register_keras_serializable()
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


def residual_block(x, units, dropout_rate=0.3):
    # Shortcut connection
    residual = x
    if residual.shape[-1] != units:
        residual = Dense(units, kernel_regularizer=l2(0.01))(residual)

    x = Dense(units, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(units, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x


@keras.saving.register_keras_serializable()
class SaveFirstNumber(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved_number = None

    def build(self, input_shape):
        self.saved_number = self.add_weight(
            shape=(), 
            initializer='zeros', 
            trainable=False,
            name='saved_number'
        )

    def call(self, inputs):
        self.saved_number.assign(tf.reduce_mean(inputs[:, 0:1]))
        return inputs

    def get_config(self):
        config = super().get_config()
        return config


@keras.saving.register_keras_serializable()
class MultiplyBySavedNumber(Layer):
    def __init__(self, saved_number_layer, **kwargs):
        super(MultiplyBySavedNumber, self).__init__(**kwargs)
        self.saved_number_layer = saved_number_layer

    def call(self, inputs):
        saved_number = self.saved_number_layer.saved_number
        return inputs * saved_number

    def get_config(self):
        config = super().get_config()
        config.update({
            'saved_number_layer': keras.saving.serialize_keras_object(self.saved_number_layer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['saved_number_layer'] = keras.saving.deserialize_keras_object(config['saved_number_layer'])
        return cls(**config)
    
def build_complex_rnac_model(experiment_id, input_shape, compile=True):
    inputs = Input(shape=input_shape)
    
    # save the pearson correlation HTS model result over the test set
    save_first_number_layer = SaveFirstNumber()
    x = save_first_number_layer(inputs)
    
    # Main branch
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Residual blocks
    x = residual_block(x, 256)
    x = residual_block(x, 128)
    
    # Secondary branch
    y = Dense(256, kernel_regularizer=l2(0.01))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.3)(y)
    y = Dense(128, kernel_regularizer=l2(0.01))(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    # Concatenate branches
    combined = Concatenate()([x, y])
    
    # Final layers
    z = Dense(128, kernel_regularizer=l2(0.01))(combined)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Dropout(0.3)(z)
    
    # Multiply by the saved number (pearson correlation)
    z = MultiplyBySavedNumber(save_first_number_layer)(z)
    
    z = Dense(64, kernel_regularizer=l2(0.01))(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    outputs = Dense(1, activation='linear')(z)
    
    model = Model(inputs=inputs, outputs=outputs, name=experiment_id)
    
    if compile:
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss=MeanSquaredError())
    
    return model

def load_complex_model(model_path: str)-> Model:
    custom_objects = {
    'SaveFirstNumber': SaveFirstNumber,
    'MultiplyBySavedNumber': MultiplyBySavedNumber
    }
    model = load_model(model_path, custom_objects=custom_objects)
    return model

def load_HTS_to_RNAC_scaler(path) -> StandardScaler:
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

### TRAIN RNAC TO RNAC MODEL ###

def train_rnac_model(model, X, y, X_test=[], y_test=[],
                      batch_size=256, epochs=100, test_size=0.2,patience=5):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
    if not len(X_test):
        # Create test set from the validation set
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.10, random_state=42)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True) 
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    # Evaluate the model on the validation set
    val_loss = model.evaluate(X_val, y_val)
    eval_dict = {'val_loss': val_loss}

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    eval_dict['test_loss'] = test_loss
    print(f"Validation Loss: {val_loss}")
    return model, history, eval_dict

def scale_train_data(model_id, X, save_dir):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # save the scaler for further user
    os.makedirs(save_dir, exist_ok=True)
    scaler_file = f'{save_dir}/{model_id}_scaler.pkl'
    # picke the scaler
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    return X_scaled

def train_complex_model(model_id,X,y, X_test, y_test, save_dir):
    # scale the data
    X_scaled = scale_train_data(model_id, X, save_dir)
    model = build_complex_rnac_model(model_id, X.shape[1:])
    model, history, eval_dict = train_rnac_model(model, X, y, X_test, y_test)
    save_model_path = f'{save_dir}/{model_id}.keras'
    save_model(model, save_model_path)
    print(f'Model saved at {save_model_path}')
    # save the history and eval dict as one json file in the save_dir
    result_dict = {'history': history.history}
    result_dict.update(eval_dict)
    history_file = f'{save_dir}/{model_id}_history.json'
    return model, history

def save_RNAC_preds(prediction: np.ndarray, protein_idx:int):
    saving_dir = 'data/final/RNAC_3_model_predictions'
    os.makedirs(saving_dir, exist_ok=True)
    prediction_df = pd.DataFrame(prediction, columns=HTS_TO_RNAC_COLS)
    prediction_df.to_csv(f'{saving_dir}/RBP{protein_idx}.csv')

def save_RNAC_final_results(prediction: np.ndarray, protein_idx:int):
    saving_dir = 'data/final/RNAC_final_predictions'
    os.makedirs(saving_dir, exist_ok=True)
    with open(f'{saving_dir}/RBP{protein_idx}.txt', 'w') as f:
        # write only numbers, one in each line
        f.write('\n'.join(prediction.flatten().astype(str)))
 
def load_HTS_prediction(protein_idx:int)->np.ndarray:
    prediction_path = f'data/final/train_predictions/RBP{protein_idx}.csv'
    prediction = pd.read_csv(prediction_path, index_col=0)
    return prediction[COL_ORDER].values

def get_HTS_to_RNAC_prediction(HTS_prediction:pd.DataFrame)->np.ndarray:
    HTS_prediction = HTS_prediction[COL_ORDER].values
    model = load_complex_model(COMPLEX_BIG_MODEL_PATH)
    scaler = load_HTS_to_RNAC_scaler(COMPLEX_BIG_MODEL_SCALER_PATH)
    X_scaled = scaler.transform(HTS_prediction)
    predictions = model.predict(X_scaled)
    return predictions
    