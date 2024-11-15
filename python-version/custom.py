import train
import inference
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

def load_model(code_dir):
    model_path = os.path.join(code_dir, "python_model.pkl")
    return train.load_model(model_path)

def score(data, model, **kwargs):
    print("================")
    print(kwargs)
    print("================")
    data = train.feature_engineering(data)
    X = train.create_X(data)
    preds = model.predict(X)
    return pd.DataFrame(data= preds, columns=["Predictions"])

def fit(X: pd.DataFrame, y: pd.Series, parameters=None, **kwargs) -> None:
    print("================")
    print(kwargs)
    print("================")
    # fit a DecisionTreeRegressor
    estimator = RandomForestRegressor(
        n_estimators=parameters["n_estimators"], max_depth=parameters["max_depth"], random_state=parameters["random_state"]
    )
    X = train.feature_engineering(X)
    X = train.create_X(X)
    estimator.fit(X, y)
    with open( os.path.join( kwargs["output_dir"], "python_model.pkl"), "wb") as f:
        f.write(pickle.dumps(estimator))
