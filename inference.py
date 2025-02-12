import torch
import torchaudio
import audiocraft.models
from audiocraft.utils.notebook import display_audio

# Define checkpoint path
CHECKPOINT_DIR = "/workspace/audiocraft/checkpoints/my_audio_lm/"

# Load fine-tuned model
print("🚀 Loading fine-tuned MAGNeT model...")
magnet = audiocraft.models.MAGNeT.get_pretrained(CHECKPOINT_DIR)

print("✅ Model loaded successfully!")

# Define text prompt for music generation
prompt = "Energetic upbeat koplo music with strong kendang and bass, suitable for dancing."

# Generate audio (MAGNeT will handle device placement)
print("🎵 Generating music...")
output = magnet.generate([prompt], progress=True)  # Ensure the prompt is wrapped in a list

# Move generated tensor to CPU before saving
audio_tensor = output[0].cpu()

# Save audio file
OUTPUT_PATH = "/workspace/audiocraft/generated_music.wav"
torchaudio.save(OUTPUT_PATH, audio_tensor, sample_rate=32000)

print(f"✅ Audio file saved at: {OUTPUT_PATH}")

# Play audio (if running in a Jupyter notebook)
display_audio(audio_tensor, sample_rate=32000)
