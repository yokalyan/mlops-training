
import requests

url = "http://localhost:9696/predict"
ride = {
    'PULocationID': 10,
    'DOLocationID': 50,
    'trip_distance': 50 
}

response = requests.post(url, json=ride)

if response.status_code == 200:
    prediction = response.json()
    print(f"Predicted duration: {prediction['duration']:.2f} seconds")
    print(f"Run ID: {prediction['RUN_ID']}")
    print(f"MLflow Tracking URI: {prediction['MLFLOW_TRACKING_URI']}")
else:
    print(f"Error: {response.status_code}, {response.text}")

