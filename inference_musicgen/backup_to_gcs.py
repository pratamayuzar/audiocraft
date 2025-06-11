import os
import base64
import tempfile
from google.cloud import storage

def save_key_from_env(env_var="GCP_KEY_B64"):
    """Decode base64-encoded GCP key from environment and write to temp file."""
    key_b64 = os.getenv(env_var)
    if not key_b64:
        raise ValueError("Missing GCP_KEY_B64 environment variable.")

    decoded = base64.b64decode(key_b64)
    temp_key_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_key_path.write(decoded)
    temp_key_path.close()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path.name
    return temp_key_path.name

def upload_folder_to_gcs(bucket_name, source_folder, destination_blob_prefix="backup"):
    """Recursively uploads a folder to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(source_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_folder)
            blob_path = f"{destination_blob_prefix}/{relative_path}"

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded {local_path} → gs://{bucket_name}/{blob_path}")

if __name__ == "__main__":
    # 🔧 Adjust these values
    GCS_BUCKET = "finetuning-dataset"
    LOCAL_MODEL_FOLDER = "/workspace/audiocraft/checkpoints/my_musicgen_model_no_vocal_250319"
    GCS_FOLDER_PREFIX = "checkpoint/musicgen-koplo"

    key_path = save_key_from_env()  # Decode and write service key

    upload_folder_to_gcs(GCS_BUCKET, LOCAL_MODEL_FOLDER, GCS_FOLDER_PREFIX)

    os.remove(key_path)  # Optional: clean up the key file


# export GCP_KEY_B64="PASTE_YOUR_BASE64_ENCODED_SERVICE_ACCOUNT_KEY"
# python backup_to_gcs.py