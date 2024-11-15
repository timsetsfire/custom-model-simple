"""
Train the ML model that will be used during inference.
This implementation specifically downloads the insurance dataset, which comes with the
Machine Learning with R book (https://github.com/stedy/Machine-Learning-with-R-datasets).
"""

import requests
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import time
import os

# path to where the model will be saved

MODEL_PATH = "python_model.pkl"


def load_training_data() -> pd.DataFrame:
    """
    Accesses the dataset and returns a pandas DataFrame that can be used for model training.
    """
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    r = requests.get(url)
    table = r.text
    df = pd.read_csv(StringIO(table))
    df.to_csv("training_data.csv", index = False)
    # train-test split for training set
    training_df = df.iloc[: int(0.8 * len(df)), :]
    return training_df


def feature_engineering(df) -> pd.DataFrame:
    """
    Take the raw input dataframe and return a dataframe with new features.
    Return dataframe shape: (subjects, features)
    """
    ## convert binary features (smoke, sex) to numeric
    df["smoker"] = np.where(df["smoker"] == "yes", 1, 0)
    df["sex"] = np.where(df["sex"] == "male", 1, 0)
    ## convert categorical features to one hot encoding
    df["region"] = pd.Categorical(
        df["region"], categories=["southeast", "southwest", "northeast", "northwest"]
    )
    region_dummies = pd.get_dummies(df["region"])
    df = df.drop("region", axis=1)
    df = pd.concat([df, region_dummies], axis=1)
    ## do some manipulation on numerical data
    df["children_squared"] = df["children"] ** 2
    df["bmi_log"] = np.log(df["bmi"])
    ## not sure if any of this makes sense, but just for testing purposes
    return df


def create_X(df) -> np.ndarray:
    """
    Take the raw input dataframe and return a feature matrix (numpy array) that can be used for training.
    Return array shape: (subjects, features)
    """
    features = [col for col in df.columns if col != "charges"]
    X = df[features].values
    return X


def create_y(df) -> np.ndarray:
    """
    Take the raw input dataframe and return a target vector (numpy array) that can be predicted.
    Return array shape: (subjects, 1)
    """
    y = df["charges"].values
    return y


def train_model(X, y) -> RandomForestRegressor:
    """
    Train the model that will be used for inference.
    """
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    return model


def save_model(model, path=MODEL_PATH) -> None:
    """
    Save the (production ready) model that has been trained.
    """
    # Configure S3 client
    # This seems not a very secure practice
    # access_key = os.environ["IAM_ACCESS_KEY"]
    # secret_key = os.environ["IAM_SECRET_KEY"]
    # session = boto3.Session(
    #     aws_access_key_id=access_key, aws_secret_access_key=secret_key
    # )
    # s3 = session.client("s3")
    # model_binary = pickle.dumps(model)
    # s3.put_object(
    #     Bucket="mlops-unit-test-saved-models", Key=MODEL_PATH, Body=model_binary
    # )
    with open(path, "wb") as f:
        f.write(pickle.dumps(model))


def load_model(path=MODEL_PATH) -> RandomForestRegressor:
    """
    Load the trained model
    """
    # Configure S3 client
    # This seems not a very secure practice
    # access_key = os.environ["IAM_ACCESS_KEY"]
    # secret_key = os.environ["IAM_SECRET_KEY"]
    # session = boto3.Session(
    #     aws_access_key_id=access_key, aws_secret_access_key=secret_key
    # )
    # s3 = session.client("s3")
    # model_binary = s3.get_object(Bucket="mlops-unit-test-saved-models", Key=MODEL_PATH)[
    #     "Body"
    # ].read()
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    # try:
    start_time = time.time()
    df = load_training_data()
    df = feature_engineering(df)
    X = create_X(df)
    y = create_y(df)
    model = train_model(X, y)
    save_model(model)
    train_time = f"Training time: {time.time() - start_time} seconds"
    path = f"Model saved at: {MODEL_PATH}"
    return {"Status": "Success", "msg 1": train_time, "msg 2": path}
    # except:
    #     return {"Status" : "Error"}


if __name__ == "__main__":
    main()
