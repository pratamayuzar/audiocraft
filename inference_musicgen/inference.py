import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio

# Load the exported fine-tuned model
CHECKPOINT_PATH = "/workspace/audiocraft/exported/musicgen_base_250312.pt"
print("🚀 Loading fine-tuned MusicGen model...")
musicgen = MusicGen.get_pretrained(CHECKPOINT_PATH)
print("✅ Model loaded successfully!")

# Define multiple text prompts
prompts = [
    "Energetic upbeat koplo music with strong kendang and bass, suitable for dancing.",
    "A lively koplo track with driving kendang rhythms, cheerful melodies on the keyboard, and strong, punchy percussion."
]

# Generate audio from prompts
print("🎵 Generating music...")
output = musicgen.generate(prompts, progress=True)  # Generate music for both prompts

# Save generated audio files
for i, audio_tensor in enumerate(output):
    audio_tensor = audio_tensor.cpu()  # Move to CPU before saving
    output_path = f"/workspace/audiocraft/inference/generated_music_{i+1}.wav"
    torchaudio.save(output_path, audio_tensor, sample_rate=32000)
    print(f"✅ Audio file saved at: {output_path}")
