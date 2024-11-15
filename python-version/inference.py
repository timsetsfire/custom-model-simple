"""
API for using the model trained in `train.py`.
"""
 
from typing import Literal
 
from train import load_model, create_X, create_y, feature_engineering
import requests
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import random
import time
 
from typing import Literal, Union
# path to where the model will be saved
MODEL_PATH = 'python_model.pkl'
INFERENCE_MODE = 'batch'
PREDICTION_PATH = 'predictions.txt'
 
def load_testing_data() -> pd.DataFrame:
    '''
    Accesses the dataset and returns a pandas DataFrame that can be used for model training.
    '''
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    r = requests.get(url)
    table = r.text
    df = pd.read_csv(StringIO(table))
   
    # train-test split for training set
    training_df = df.iloc[int(0.8*len(df)):, :]
    return training_df
 
 
def batch_inference(model, X) -> np.ndarray:
    '''
    Take in new features and return a prediction (charges)
    '''
 
    predictions = model.predict(X)
    return predictions
 
def batch_inference(model, X) -> np.ndarray:
    '''
    Take in new features and return a prediction (charges)
    '''
 
    predictions = model.predict(X)
    return predictions
 
def evaluation(predictions, y) -> None:
    print('MAE:', mean_absolute_error(y, predictions))
    print('MSE:', mean_squared_error(y, predictions))
 
   
def inference(model, X, y) -> np.ndarray:
   
    index = random.randint(0,X.shape[1])
    prediction = model.predict(X[index].reshape(1, -1))
    return prediction
 
def save_predictions(predictions, path=PREDICTION_PATH) -> None:
    np.savetxt(PREDICTION_PATH, predictions, fmt='%d')
   
    
def main():
    start_time = time.time()
    df = load_testing_data()
    df = feature_engineering(df)
    X = create_X(df)
    y = create_y(df)
    model = load_model()
    if INFERENCE_MODE == "batch":
        predictions = batch_inference(model, X)
        evaluation(predictions, y)
    else:
        predictions = inference(model, X, y)
    save_predictions(predictions)
    inference_time = f"Inference time: {time.time() - start_time} seconds"
    path = f"Prediction saved at: {PREDICTION_PATH}"
    return {"Status" : "Success", "msg 1" : inference_time, "msg 2" : path}
 
if __name__ == '__main__':
    main()
 