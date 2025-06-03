
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
else:
    print(f"Error: {response.status_code}, {response.text}")

