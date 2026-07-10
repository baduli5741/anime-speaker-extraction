import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Setup encoding for safe Windows stdout prints
sys.stdout.reconfigure(encoding='utf-8')

# Ensure we use GPU if CUDA is available in our current environment (ai_env)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Attempt to load Silero VAD
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    print("Silero VAD model loaded successfully.")
except Exception as e:
    print(f"Error loading Silero VAD model: {e}")
    sys.exit(1)

def vad_slice_audio(input_wav_path, output_dir, file_prefix="slice", threshold=0.4, min_silence_duration_ms=300):
    """
    Load an audio file, apply Silero VAD to detect speech segments, and slice the audio accordingly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing: {input_wav_path}")
    
    # Read audio (Silero expects 16kHz)
    try:
        wav = read_audio(input_wav_path, sampling_rate=16000)
    except Exception as e:
        print(f"Error reading audio file {input_wav_path}: {e}")
        return []
    
    # Detect speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=16000,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms
    )
    
    print(f"Detected {len(speech_timestamps)} speech segments.")
    
    sliced_files = []
    
    for idx, segment in enumerate(speech_timestamps):
        start_sample = segment['start']
        end_sample = segment['end']
        
        # Crop the segment
        segment_wav = wav[start_sample:end_sample]
        
        # Save output
        out_filename = f"{file_prefix}_{idx+1:03d}.wav"
        out_filepath = output_path / out_filename
        
        # Save to disk using torchaudio (since silero save_audio uses torchaudio)
        try:
            torchaudio.save(str(out_filepath), segment_wav.unsqueeze(0), 16000)
            sliced_files.append(out_filepath)
            duration = (end_sample - start_sample) / 16000.0
            print(f"  -> Saved segment {idx+1:03d}: {out_filename} ({duration:.2f}s, start={start_sample/16000.0:.2f}s, end={end_sample/16000.0:.2f}s)")
        except Exception as e:
            print(f"Error saving segment {idx+1}: {e}")
            
    return sliced_files

if __name__ == "__main__":
    # Test on a generated sample or any raw audio file
    # We will look for an existing wav file to perform a proof of concept
    test_src = r"C:\Users\piai\Desktop\new_anti\voicebox-main\voicebox-main\output\frieren_1_sample_korean_test_poc.wav"
    test_out_dir = r"C:\Users\piai\Desktop\new_anti\sliced_samples_poc"
    
    if os.path.exists(test_src):
        vad_slice_audio(test_src, test_out_dir, file_prefix="frieren_vad_poc")
    else:
        print(f"Source file not found at: {test_src}")
