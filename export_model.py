import os
from audiocraft.utils import export

# Define paths
CHECKPOINT_PATH = "/workspace/trained_model/vocal/xps/e1cd28f5/checkpoint.th"
EXPORT_DIR = "/workspace/audiocraft/checkpoints/my_audio_lm/"
ENCODEC_MODEL = "facebook/encodec_32khz"  # Change if using a custom EnCodec model

# Ensure export directory exists
os.makedirs(EXPORT_DIR, exist_ok=True)

# Export the fine-tuned MAGNeT model
print("🚀 Exporting fine-tuned MAGNeT model...")
export.export_lm(CHECKPOINT_PATH, os.path.join(EXPORT_DIR, "state_dict.bin"))

# Export the EnCodec model (pretrained)
print("🎼 Exporting EnCodec model...")
export.export_pretrained_compression_model(
    ENCODEC_MODEL,
    os.path.join(EXPORT_DIR, "compression_state_dict.bin")
)

print("✅ Model export completed!")
print(f"📂 Files saved to: {EXPORT_DIR}")
