import os
import torch
import torchaudio
from audiocraft.models import MusicGen

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
musicgen.set_generation_params(duration=60)  # ✅ Correct way to set duration

# Generate audio from prompts
print(f"🎵 Generating {musicgen.duration}-second music...")
output = musicgen.generate(prompts, progress=True)  # No duration argument here

# Save generated audio files
OUTPUT_DIR = "/workspace/audiocraft/inference_musicgen/test/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, audio_tensor in enumerate(output):
    audio_tensor = audio_tensor.cpu()  # Move to CPU before saving
    output_path = f"{OUTPUT_DIR}generated_music_{i+1}.wav"
    torchaudio.save(output_path, audio_tensor, sample_rate=32000)
    print(f"✅ Audio file saved at: {output_path}")
