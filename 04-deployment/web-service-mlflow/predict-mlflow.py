import pickle
import mlflow
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
RUN_ID = 'ea5d80a75c6548f7a42d857a6d412935'

#
# This version relies on tracking URI and run ID
# One way to avoid using the tracking URI is to load the load
# model on S3, and then load it directly from S3. We will fetch the RUN_ID from the S3 bucket.
# Alternatively, we can use the tracking URI and run ID to load the model.  
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("green-taxi-experiment")
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
# print("Downloaded dict_vectorizer from run_id:", RUN_ID, "to path:", path)
# with open(path, 'rb') as f_out:
#     dv = pickle.load(f_out)
    
app = Flask("deployment-webservice")

#with open('lin_reg.bin', 'rb') as f_in:
#    dv, model = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    # X = dv.transform(features) 
    # y_pred = model.predict(X)
    y_pred = model.predict(features)
    return float(y_pred[0])

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.json
    #ride = request.get_json()
    features = prepare_features(ride)
    prediction = predict([features])
    result = {'duration': prediction,
              'RUN_ID': RUN_ID,
              'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI}
    
    #return {'duration': prediction}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9696, debug=True)
    