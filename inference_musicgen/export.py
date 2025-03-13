import os
import torch
from audiocraft.models import MusicGen

# Define paths
CHECKPOINT_PATH = "/workspace/trained_model/no_vocal_250312/xps/a787bbb9/checkpoint.th"
EXPORT_DIR = "/workspace/audiocraft/exported"
EXPORT_PATH = os.path.join(EXPORT_DIR, "musicgen_base_250312.pth")

print("✅ Model export start!")

# Ensure the export directory exists
os.makedirs(EXPORT_DIR, exist_ok=True)

# Load the trained model from checkpoint
print("🚀 Loading trained MusicGen model from checkpoint...")
model = MusicGen.get_pretrained(CHECKPOINT_PATH)

# Save the model in the required format
print(f"💾 Saving exported model to {EXPORT_PATH} ...")
torch.save(model.state_dict(), EXPORT_PATH)

print("✅ Model export completed successfully!")
print(f"📂 Files saved to: {EXPORT_DIR}")