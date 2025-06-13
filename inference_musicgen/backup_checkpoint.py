import os
import tarfile
import base64
import tempfile
from google.cloud import storage

# === CONFIGURATION ===
LOCAL_MODEL_PATH = "/workspace/trained_model/no_vocal_250319/xps/3f71076c/checkpoint.th"
ARCHIVE_FILENAME = "checkpoint_backup.tar.gz"
ARCHIVE_PATH = f"/tmp/{ARCHIVE_FILENAME}"

GCS_BUCKET_NAME = "finetuning-dataset"
GCS_DESTINATION_PATH = f"checkpoint/musicgen-koplo/{ARCHIVE_FILENAME}"

# === BASE64-ENCODED GCP CREDENTIALS ===
BASE64_GCP_CREDENTIALS = os.environ.get("GCP_KEY_B64")

if not BASE64_GCP_CREDENTIALS:
    raise ValueError("❌ Missing environment variable: GCP_CREDS_BASE64")

# === DECODE AND SAVE CREDENTIALS ===
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    cred_path = tmp.name
    tmp.write(base64.b64decode(BASE64_GCP_CREDENTIALS))
    tmp.flush()

# Set GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
print(f"🔐 GCP credentials written to: {cred_path}")

# === COMPRESS ===
print(f"📦 Compressing {LOCAL_MODEL_PATH} → {ARCHIVE_PATH}")
with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
    tar.add(LOCAL_MODEL_PATH, arcname=os.path.basename(LOCAL_MODEL_PATH))
print("✅ Compression complete!")

# === UPLOAD TO GCS ===
print(f"☁️ Uploading to GCS bucket: {GCS_BUCKET_NAME} (archive tier)")
client = storage.Client()
bucket = client.bucket(GCS_BUCKET_NAME)
blob = bucket.blob(GCS_DESTINATION_PATH)

# Set storage class to ARCHIVE
blob.storage_class = "ARCHIVE"
blob.upload_from_filename(ARCHIVE_PATH)
print(f"✅ Upload complete: gs://{GCS_BUCKET_NAME}/{GCS_DESTINATION_PATH}")

# === CLEANUP ===
os.remove(cred_path)
print("🧹 Temporary credentials file removed.")
