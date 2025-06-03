#!/usr/bin/env python
# coding: utf-8
from datetime import datetime, timedelta
from time import sleep
import pickle
import argparse     
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import vstack  # Used to stack sparse matrices
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
    
from datetime import date
from prefect import task, flow
from prefect.tasks import task_input_hash
from prefect.artifacts import create_markdown_artifact, create_link_artifact

# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment-june2")

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
    print("Start again")
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    print(f"Total number of records loaded {df.shape[0]} for month {month}")
          
    # Filter out rows with missing values in the pickup and dropoff datetime columns
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    
    # Filter out trips with duration less than 1 minute or more than 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    print(f"Total number of records after filtering {df.shape[0]} for month {month}")
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

# @task(
#     name="Create the DV - check for cached version",
#     cache_key_fn=task_input_hash, 
#     cache_expiration=timedelta(days=1)
# )
def create_X(df, dv=None):
    print("Fit the dict vectorizer and create the design matrix")
    chunk_size=10000
    categorical = ['PULocationID', 'DOLocationID']
    
    # 2. If DictVectorizer isn't passed, create and fit one
    if dv is None:
        #sample_dicts = df[categorical].iloc[:chunk_size].to_dict(orient='records')
        sample_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer(sparse=True)
        dv.fit(sample_dicts)

    # 3. Initialize a list to hold each chunk's sparse matrix
    all_chunks = []
    # 4. Loop through the DataFrame in chunks
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]

        # Convert the chunk to list-of-dicts (row-wise) for DictVectorizer
        dicts = chunk[categorical].to_dict(orient='records')

        # Use the already-fitted DictVectorizer to transform
        X_chunk = dv.transform(dicts)

        # Store the sparse matrix
        all_chunks.append(X_chunk)

    # 5. Combine all sparse matrices into one using vstack
    X = vstack(all_chunks)

    # 6. Return the final design matrix and the vectorizer
    return X, dv

# Not going to use this version as it is crashing on my machine because of memory issues as
# it tries to load and transform the entire dataset into memory at once. 
# def create_X(df, dv=None):
#     categorical = ['PULocationID', 'DOLocationID']
#     numerical = ['trip_distance']
    
#     dicts = df[categorical+numerical].to_dict(orient='records')
#     if dv is None:
#         dv = DictVectorizer(sparse=True)
#         print(dicts[:5])
#         X = dv.fit_transform(dicts)
#     else:
#         X = dv.transform(dicts)
#     print("Returning from create_X")
#     return X, dv

@task(name="train_linear_regression_model")
def train_linear_regression_model(X_train, y_train, X_val, y_val, dv):
    print("Training linear regression model")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        y_pred = lr_model.predict(X_val)
       
        print("Model intercept:", lr_model.intercept_)
        mlflow.log_metric("intercept", lr_model.intercept_)
        
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        print(f"RMSE: {rmse}")

        with open('models/lin_reg.bin', 'wb') as f_out:
            pickle.dump((dv, lr_model), f_out)
        
        mlflow.log_artifact("models/lin_reg.bin", artifact_path="models")
        
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
        mlflow.sklearn.log_model(
            lr_model,
            artifact_path="model",
            registered_model_name="nyc-taxi-duration-model"
        )
        mlflow.log_param("run_id", run.info.run_id)
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

    run_id = train_linear_regression_model(X_train, y_train, X_val, y_val, dv)
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