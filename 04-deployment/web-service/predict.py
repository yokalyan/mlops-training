import pickle
from flask import Flask, request, jsonify

app = Flask("deployment-webservice")


with open('lin_reg.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    #features['PU_DO'] = f"{pulocationid}_{dolocationid}"
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features) 
    y_pred = model.predict(X)
    return float(y_pred[0])

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.json
    #ride = request.get_json()
    features = prepare_features(ride)
    prediction = predict([features])
    result = {'duration': prediction}
    #return {'duration': prediction}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9696, debug=True)
    