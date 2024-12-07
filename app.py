from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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
