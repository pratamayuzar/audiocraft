import torch
import torchaudio
import os
from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio

# Load the exported fine-tuned model
CHECKPOINT_PATH = "/workspace/audiocraft/checkpoints/my_musicgen_model_no_vocal_250312/"
print("🚀 Loading fine-tuned MusicGen model...")
musicgen = MusicGen.get_pretrained(CHECKPOINT_PATH)
print("✅ Model loaded successfully!")

# Define multiple text prompts
prompts = [
    "Energetic upbeat koplo music with strong kendang and bass, suitable for dancing.",
    "A lively koplo track with driving kendang rhythms, cheerful melodies on the keyboard, and strong, punchy percussion."
]

# Set custom duration (e.g., 60 seconds)
duration = 60  # Increase duration as needed

# Generate audio from prompts
print(f"🎵 Generating {duration}-second music...")
output = musicgen.generate(prompts, duration=duration, progress=True)  # ✅ Set duration

# Save generated audio files
OUTPUT_DIR = "/workspace/audiocraft/inference_musicgen/test/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, audio_tensor in enumerate(output):
    audio_tensor = audio_tensor.cpu()  # Move to CPU before saving
    output_path = f"{OUTPUT_DIR}generated_music_{i+1}.wav"
    torchaudio.save(output_path, audio_tensor, sample_rate=32000)
    print(f"✅ Audio file saved at: {output_path}")
