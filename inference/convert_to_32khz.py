import os
import torchaudio
from torchaudio.transforms import Resample

def convert_audio_to_32khz(input_folder, output_folder):
    """Converts all audio files in input_folder from 44.1kHz to 32kHz and saves them in output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav") or file_name.endswith(".mp3"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            # Load the audio file
            waveform, sample_rate = torchaudio.load(input_path)
            
            # Only resample if it's 44.1kHz
            if sample_rate == 44100:
                resampler = Resample(orig_freq=44100, new_freq=32000)
                waveform = resampler(waveform)
                
                # Save the converted file
                torchaudio.save(output_path, waveform, 32000)
                print(f"✅ Converted: {file_name} -> 32kHz")
            else:
                print(f"⚠️ Skipping {file_name}, already {sample_rate}Hz")

# Define input and output folders
input_folder = "dataset/lagoe_44"
output_folder = "dataset/lagoe"

# Run conversion
convert_audio_to_32khz(input_folder, output_folder)
