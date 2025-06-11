import os
from audiocraft.utils import export

# Define paths
CHECKPOINT_PATH = "/workspace/trained_model/no_vocal_250319/xps/3f71076c/checkpoint.th"
EXPORT_DIR = "/workspace/audiocraft/checkpoints/my_musicgen_model_no_vocal_250319/"
ENCODEC_MODEL = "facebook/encodec_32khz"  # Change if using a custom EnCodec model

# Ensure export directory exists
os.makedirs(EXPORT_DIR, exist_ok=True)

# Export the fine-tuned MusicGen model
print(f"🚀 Exporting fine-tuned MusicGen model from: {CHECKPOINT_PATH}")
export.export_lm(CHECKPOINT_PATH, os.path.join(EXPORT_DIR, "state_dict.bin"))  # ✅ FIXED FUNCTION

# Export the EnCodec model (pretrained)
print("🎼 Exporting EnCodec model...")
export.export_pretrained_compression_model(
    ENCODEC_MODEL,
    os.path.join(EXPORT_DIR, "compression_state_dict.bin")
)

print("✅ Model export completed!")
print(f"📂 Files saved to: {EXPORT_DIR}")
