import torch
import torchaudio
import audiocraft.models
from audiocraft.utils.notebook import display_audio

# Define checkpoint path
CHECKPOINT_DIR = "/workspace/audiocraft/checkpoints/my_audio_lm/"

# Load your fine-tuned model
print("🚀 Loading fine-tuned MAGNeT model...")
magnet = audiocraft.models.MAGNeT.get_pretrained(CHECKPOINT_DIR)

# Move model to GPU if available
if torch.cuda.is_available():
    magnet.cuda()  # ✅ Fix: Move model to GPU properly
    device = "cuda"
else:
    magnet.cpu()
    device = "cpu"

print(f"✅ Model loaded on {device}")

# Define text prompt for music generation
prompt = "Energetic upbeat koplo music with strong kendang and bass, suitable for dancing."

# Generate audio
print("🎵 Generating music...")
output = magnet.generate(prompt, progress=True)  # Ensure generate() works

# Save audio file
OUTPUT_PATH = "/workspace/audiocraft/generated_music.wav"
torchaudio.save(OUTPUT_PATH, output[0].cpu(), sample_rate=32000)

print(f"✅ Audio file saved at: {OUTPUT_PATH}")

# Play audio (if running in a Jupyter notebook)
display_audio(output[0], sample_rate=32000)
