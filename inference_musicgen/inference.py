import os
import torch
import torchaudio
from audiocraft.models import MusicGen

# Load the exported fine-tuned model
CHECKPOINT_PATH = "/workspace/audiocraft/checkpoints/my_musicgen_model_no_vocal_250319/"
print("🚀 Loading fine-tuned MusicGen model...")
musicgen = MusicGen.get_pretrained(CHECKPOINT_PATH)
print("✅ Model loaded successfully!")

# Define multiple text prompts
prompts = [
    "Energetic upbeat koplo music with strong kendang and bass, suitable for dancing.",
    "A lively koplo track with driving kendang rhythms, cheerful melodies on the keyboard, and strong, punchy percussion.",
    "A dynamic Koplo composition blending traditional elements with electronic beats, featuring sharp kendang accents, deep bass, and vibrant brass for an exciting fusion sound.",
    "A mid-tempo, smooth Koplo tune featuring soft kendang rhythms, melodic flute sections, and warm keyboard harmonies, evoking a relaxed yet emotional mood.",
    "A laid-back, groove-driven Koplo song with deep, rolling kendang, soulful electric guitar licks, and smooth synth pads, perfect for a chilled yet rhythmic listening experience.",
]

# Set custom duration (e.g., 60 seconds)
musicgen.set_generation_params(duration=60)  # ✅ Correct way to set duration

# Generate audio from prompts
print(f"🎵 Generating {musicgen.duration}-second music...")
output = musicgen.generate(prompts, progress=True)  # No duration argument here

# Save generated audio files
OUTPUT_DIR = "/workspace/audiocraft/inference_musicgen/audio-250319/new/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, audio_tensor in enumerate(output):
    audio_tensor = audio_tensor.cpu()  # Move to CPU before saving
    output_path = f"{OUTPUT_DIR}generated_music_{i+1}.wav"
    torchaudio.save(output_path, audio_tensor, sample_rate=32000)
    print(f"✅ Audio file saved at: {output_path}")
