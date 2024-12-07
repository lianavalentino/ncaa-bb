from flask import Flask, request, jsonify
import pandas as pd
import os
from google.cloud import storage
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

app = Flask(__name__)

# Set up Google Cloud Storage client
BUCKET_NAME = "ncaa_bb"  # Your GCS bucket name
TRAINING_FILE = "data.csv"  # Path to training data in GCS
MODEL_FILE = "model.pkl"  # Path to save/load the model in GCS


def fetch_training_data():
    """Download the training data from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(TRAINING_FILE)
    temp_file = "/tmp/training_data.csv"
    blob.download_to_filename(temp_file)
    return pd.read_csv(temp_file)


def upload_model_to_gcs(local_path, gcs_path):
    """Upload the trained model to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)


def download_model_from_gcs():
    """Download the trained model from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)
    temp_model_path = "/tmp/" + MODEL_FILE
    blob.download_to_filename(temp_model_path)
    return temp_model_path


@app.route('/train', methods=['POST'])
def train_model():
    """Train a logistic regression model and save it to Google Cloud Storage."""
    try:
        # Step 1: Fetch data
        data = fetch_training_data()
        data.dropna(inplace=True)
        
        # Step 2: Load the dataset
        print("Loading dataset...")
        data = pd.read_csv(local_data_file)
        data.dropna(inplace=True)
    
        # Filter only the features starting with h_ and a_ for training
        features = [col for col in data.columns if col.startswith("h_") or col.startswith("a_")]
        X = data[features]
        y = data["home_win"]  # Assuming "result" is the target column (0 for loss, 1 for win)
    
        # Step 3: Train the logistic regression model
        print("Training logistic regression model...")
        model = LogisticRegression()
        model.fit(X, y)
    
        # Step 4: Export the trained model to a pickle file
        local_model_file = "model.pkl"
        with open(local_model_file, "wb") as f:
            pickle.dump(model, f)
        print("Model trained and saved locally as model.pkl.")
        
        # Step 5: Upload the model to GCS
        upload_model_to_gcs(local_model_file, MODEL_FILE)
        
        return jsonify({"message": "Model training completed and uploaded to GCS"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict the result of a game based on the provided features."""
    try:
        # Step 1: Download the trained model
        model_path = download_model_from_gcs()
        model = joblib.load(model_path)

        # Step 2: Parse input data
        input_data = request.get_json()
        features = np.array([input_data['features']])
        # features = np.array([
        #     [
        #         input_data['h_eFGp'], input_data['h_FTr'], input_data['h_ORBp'], input_data['h_TOVp'],
        #         input_data['a_eFGp'], input_data['a_FTr'], input_data['a_ORBp'], input_data['a_TOVp']
        #     ]
        # ])
        
        # Step 3: Make prediction
        prediction = model.predict(features)
        result = "home_win" if prediction[0] == 1 else "away_win"
        probabilities = model.predict_proba(features)
        
        # Step 4: Return response
        return jsonify({
            "result": result,  # 1 = Home win, 0 = Away win
            "probabilities": {
                "home_win": probabilities[0][1],
                "away_win": probabilities[0][0]
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def health_check():
    return "App is running!", 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
