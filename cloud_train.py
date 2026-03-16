"""
Google Cloud / Vertex AI training wrapper.
Handles GCS data loading and model saving.
"""
import os
from train import train, CONFIG
from google.cloud import storage

def download_from_gcs(bucket_name, source_blob, dest_path):
    """Download file from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_path)
    print(f"Downloaded gs://{bucket_name}/{source_blob} to {dest_path}")

def upload_to_gcs(bucket_name, source_dir, dest_prefix):
    """Upload directory to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(source_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_dir)
            blob_path = f"{dest_prefix}/{relative_path}"

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")

def cloud_train():
    """Training entrypoint for cloud execution."""

    # Read environment variables
    bucket_name = os.getenv("GCS_BUCKET", "your-bucket-name")
    data_blob = os.getenv("GCS_DATA_PATH", "data/train_distill.json")
    output_prefix = os.getenv("GCS_OUTPUT_PREFIX", "models/distilled")

    # Download training data
    local_data_path = "train_distill.json"
    download_from_gcs(bucket_name, data_blob, local_data_path)

    # Train
    train(data_path=local_data_path, config=CONFIG)

    # Upload model to GCS
    upload_to_gcs(bucket_name, CONFIG["output_dir"], output_prefix)

    print("Cloud training complete!")

if __name__ == "__main__":
    cloud_train()
