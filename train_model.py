import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from google.cloud import storage

# Configure Google Cloud Storage bucket name
BUCKET_NAME = "ncaa_bb"
DATA_FILE = "data.csv"  # Replace with your dataset file name in the bucket
MODEL_FILE = "model.pkl"

def download_file_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

def upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to bucket {bucket_name} as {destination_blob_name}.")

def train_model():
    # Step 1: Download the dataset from GCS
    local_data_file = "local_game_data.csv"
    download_file_from_gcs(BUCKET_NAME, DATA_FILE, local_data_file)

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
    upload_file_to_gcs(BUCKET_NAME, local_model_file, MODEL_FILE)

if __name__ == "__main__":
    train_model()
