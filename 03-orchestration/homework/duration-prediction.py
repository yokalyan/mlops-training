#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
import pickle
import argparse     
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from datetime import date
from prefect import task, flow
from prefect.cache_policies import DEFAULT
from prefect.tasks import task_input_hash
from prefect.artifacts import create_markdown_artifact, create_link_artifact

# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

# Create a folder for models if it doesn't exist
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@task(name="Scheduled run, get current year and month")
def get_current_year_month():
    today = datetime.today()
    return today.year, today.month

# Define the Prefect tasks and flow
@task(name="Read dataframe", retries=3, retry_delay_seconds=5, log_prints=True)
def read_dataframe(year, month):
    # Read the parquet file for the specified year and month
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    print(f"Reading data from {url}")
    df = pd.read_parquet(url)
    
    # Filter out rows with missing values in the pickup and dropoff datetime columns
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    # Filter out trips with duration less than 1 minute or more than 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # Add a new feature 'PU_DO' that combines pickup and dropoff location IDs
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

# This code is not being used, but it is kept for reference
"""
@task(name = "Cache RMSE", cache_key_fn=task_input_hash, cache_expiration=timedelta(days=45))
def cache_rmse(rmse: float) -> float:
    return rmse
global cache_rmse1
@task(name="Compare RMSE", log_prints=True)
def compare_and_cache_rmse(new_rmse: float):
    # Re-fetch the last cached RMSE for comparison (simulate by rerunning the cache function)
    cached_rmse = cache_rmse.fn(cache_rmse1)  # Simulates using same input to get the cache
    
    if cached_rmse is None:
        print("No cached RMSE to compare with.")
        # Use a stable key for caching RMSE 
        cached = cache_rmse.submit(new_rmse).result()
    else:
        if new_rmse >= cached_rmse: 
            print(f"Made it worse, current and previous RMSE: {new_rmse:.4f}:{cached_rmse:.4f}")
        else:
            print(f"Improved it, current and previous RMSE: {new_rmse:.4f}:{cached_rmse:.4f}")
            # Use a stable key for caching RMSE 
            cached = cache_rmse.submit(new_rmse).result()
"""
            
@task(
    name="Create the DV - check for cached version",
    cache_key_fn=task_input_hash, 
    cache_expiration=timedelta(days=1)
)
def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

"""
Not sure if worth having cached version of this task, since it is expensive and
not worth running it if the input is the same
@task(
    name="Train the model, use the cached version if input is same",
    cache_key_fn=task_input_hash, 
    cache_expiration=timedelta(days=1)
)
"""
@task(name="Train the model")
def train_model(X_train, y_train, X_val, y_val, dv):
    cache_rmse1 = 10.0  # Initialize with a high value for comparison
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=5,
            evals=[(valid, 'validation')],
            early_stopping_rounds=5
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        
        markdown_artifact = f"""
            ###  XGBoost Training Model RMSE
            |**  Date   **|** Training RMSE **|
            |-------------|--------------------
            |  {date.today()} |   {rmse:.4f}          |
            """
        create_markdown_artifact(
            key="rmse-report",
            markdown=markdown_artifact,
            description="RMSE performance report for the XGBoost model trained on NYC taxi data."
        )
        
        create_link_artifact(
            key="rrmse-code-data-link-report",
            link="https://mygitrepourl.com",
            link_text="GitHub Repo:",
            description="We can use this to point to the data source or code version",
        )
        
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

# Validate parsed arguments. They can be none when running the flow by scheduler
@task(name="Validate arguments")
def validate_args(year, month):
    if (year is None) != (month is None):
        raise ValueError("You must provide both year and month together, or neither.")
    if year is not None and (year < 2020 or year > datetime.now().year):
        raise ValueError("Year must be between 2020 and current year.")
    if month is not None and (month < 1 or month > 12):
        raise ValueError("Month must be between 1 and 12")
        
@flow(
    name="Duration training flow", 
    timeout_seconds=1200, 
    retries=3, retry_delay_seconds=5, 
    log_prints=True
)
def main_flow(year, month):
    # Assign defaults if neither was provided
    if year is None and month is None:
        print("None values for year and month, using 2 and 3 months prior.")
        year, month = get_current_year_month()
        if month <= 3:
            year -= 1
        month -= 3

    validate_args(year, month)
        
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, default=None, help='Year of the data to train on (e.g. 2025)')
    parser.add_argument('--month', type=int, default=None, help='Month of the data to train on (e.g. 1-12)')
    args = parser.parse_args()
    
    run_id = main_flow(year=args.year, month=args.month)
    with open("run_id.txt", "w") as f:
        f.write(run_id)
