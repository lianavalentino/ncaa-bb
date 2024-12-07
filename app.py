from flask import Flask, request, jsonify
import pickle
import numpy as np
from google.cloud import storage

app = Flask(__name__)

# Load the model
def load_model():
    client = storage.Client()
    bucket = client.bucket("ncaa_bb")
    blob = bucket.blob("model.pkl")
    blob.download_to_filename("model.pkl")

    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    input_data = request.get_json()
    features = np.array([input_data['features']])

    # Predict
    prediction = model.predict(features)
    result = "win" if prediction[0] == 1 else "loss"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
