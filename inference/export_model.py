import torch
from audiocraft.utils import export
from audiocraft import train

# Set paths
CHECKPOINT_PATH = "/workspace/trained_model/vocal/xps/e1cd28f5/checkpoint.th"
EXPORT_DIR = "/workspace/audiocraft/checkpoints/my_audio_lm/"

# Ensure the export directory exists
import os
os.makedirs(EXPORT_DIR, exist_ok=True)

# Export the language model (MAGNeT)
print("🚀 Exporting fine-tuned MAGNeT model...")
export.export_lm(CHECKPOINT_PATH, os.path.join(EXPORT_DIR, "state_dict.bin"))
print(f"✅ Model exported to {EXPORT_DIR}/state_dict.bin")

# Export the EnCodec compression model
# If you trained your own EnCodec model:
# XP_ENCODEC_SIG = 'SIG_OF_ENCODEC'  # Replace with actual signature
# xp_encodec = train.main.get_xp_from_sig(XP_ENCODEC_SIG)
# export.export_encodec(xp_encodec.folder / 'checkpoint.th', os.path.join(EXPORT_DIR, "compression_state_dict.bin"))

# If you used a pretrained EnCodec model:
print("🚀 Exporting pretrained EnCodec compression model reference...")
export.export_pretrained_compression_model(
    "facebook/encodec_32khz",
    os.path.join(EXPORT_DIR, "compression_state_dict.bin")
)
print(f"✅ Compression model exported to {EXPORT_DIR}/compression_state_dict.bin")
