import os
import sys
import pickle
import numpy as np
import torch
import torchaudio
import tempfile
import shutil
import re
from pydub import AudioSegment
import imageio_ffmpeg

# Force UTF-8 Output
sys.stdout.reconfigure(encoding='utf-8')

# Dynamic ffmpeg path setup for Windows
ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
temp_bin_dir = os.path.join(tempfile.gettempdir(), 'ffmpeg_bin_temp')
os.makedirs(temp_bin_dir, exist_ok=True)
ffmpeg_exe = os.path.join(temp_bin_dir, 'ffmpeg.exe')
if not os.path.exists(ffmpeg_exe):
    try:
        shutil.copy2(ffmpeg_bin, ffmpeg_exe)
    except Exception as e:
        print(f"Warning: failed to copy ffmpeg binary: {e}")
os.environ["PATH"] = temp_bin_dir + os.pathsep + os.environ.get("PATH", "")
AudioSegment.converter = ffmpeg_exe

TTS_DATASET_DIR = r"D:\anime\tts_dataset"
SUBTITLE_DIR = r"C:\Users\piai\Desktop\dialect_and_survey_backup\frieren_sub_netflix"
OUTPUT_DIR = r"D:\anime\extracted_dataset"
DB_1SAMPLE_PATH = os.path.join(OUTPUT_DIR, "speaker_embeddings_db_1sample.pkl")
DB_ACCUM_PATH = os.path.join(OUTPUT_DIR, "speaker_embeddings_db_accum.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load SpeechBrain Classifier
print("Loading SpeechBrain Encoder Classifier...")
from speechbrain.inference.speaker import EncoderClassifier
sb_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

# 2. Silero VAD Loading
print("Loading Silero VAD Model...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
vad_model = vad_model.to(device)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils

def extract_emb(path):
    try:
        signal, fs = torchaudio.load(path)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000).to(device)
            signal = resampler(signal.to(device))
        else:
            signal = signal.to(device)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        with torch.no_grad():
            embeddings = sb_classifier.encode_batch(signal)
            emb = embeddings.squeeze().cpu().numpy()
        return emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else None
    except Exception as e:
        print(f"Error extracting embedding for {path}: {e}")
        return None

def is_sfx(text):
    SFX_KEYWORDS = [
        "笑い声", "歓声", "拍手", "足音", "進行音", "爆発", "効果音",
        "鳴き声", "ドア", "鐘", "風", "雨", "音楽", "♪", "～♪",
        "叫び声", "悲鳴", "銃声", "物音", "ため息", "咳", "うなり",
        "ノック", "電話", "サイレン", "轟音", "地響き", "水音",
    ]
    clean = text.strip()
    if re.match(r'^（[^）]+）$', clean):
        inner = clean[1:-1]
        for kw in SFX_KEYWORDS:
            if kw in inner:
                return True
        if re.search(r'の.*(音|声)', inner):
            return True
    if clean in ('♪～', '～♪', '♪', '♪～♪') or clean.startswith('♪') or clean.endswith('♪'):
        return True
    return False

def extract_speaker_and_text(raw_text):
    text = re.sub(r'\{\\an\d+\}', '', raw_text).strip()
    speaker = None
    match = re.match(r'^（([^）]+)）\s*(.*)', text, re.DOTALL)
    if match:
        candidate_name = match.group(1).strip()
        remaining = match.group(2).strip()
        
        if is_sfx(f'（{candidate_name}）'):
            return None, None
        if 'と' in candidate_name and len(candidate_name) > 6:
            return None, None
        if re.search(r'\d+人', candidate_name):
            return None, None
            
        speaker = candidate_name
        text = remaining if remaining else None
    
    if text:
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'<[^>]*>', '', text)
        text = text.replace('\r', '').replace('\n', ' ').strip()
        
    return speaker, text

def parse_srt(srt_path):
    if not os.path.exists(srt_path):
        return []
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    if not blocks or (len(blocks) == 1 and not blocks[0].strip()):
        blocks = content.strip().split('\r\n\r\n')
        
    segments = []
    for block in blocks:
        lines = [l.strip('\r') for l in block.strip().split('\n')]
        if not lines:
            continue
        time_line = None
        text_lines = []
        for i, line in enumerate(lines):
            if '-->' in line:
                time_line = line
                text_lines = lines[i+1:]
                break
        if not time_line or not text_lines:
            continue
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            time_line
        )
        if not time_match:
            continue
            
        sh, sm, ss, sms = map(int, time_match.groups()[:4])
        eh, em, es, ems = map(int, time_match.groups()[4:])
        start_ms = (sh * 3600 + sm * 60 + ss) * 1000 + sms
        end_ms = (eh * 3600 + em * 60 + es) * 1000 + ems
        
        raw_text = ' '.join(text_lines).strip()
        if not raw_text or is_sfx(raw_text):
            continue
            
        speaker, text = extract_speaker_and_text(raw_text)
        if speaker and text:
            segments.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker": speaker,
                "text": text
            })
    return segments

def get_local_vad_bounds(audio_segment, start_ms, end_ms, margin_ms=500):
    w_start = max(0, start_ms - margin_ms)
    w_end = min(len(audio_segment), end_ms + margin_ms)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_wav_path = f.name
    try:
        audio_segment[w_start:w_end].set_channels(1).set_frame_rate(16000).export(temp_wav_path, format="wav")
        wav = read_audio(temp_wav_path, sampling_rate=16000).to(device)
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=16000,
            threshold=0.4,
            min_speech_duration_ms=250,
            min_silence_duration_ms=250
        )
        if not speech_timestamps:
            return start_ms, end_ms
        abs_blocks = []
        for ts in speech_timestamps:
            s_abs = w_start + int((ts['start'] / 16000.0) * 1000)
            e_abs = w_start + int((ts['end'] / 16000.0) * 1000)
            abs_blocks.append((s_abs, e_abs))
            
        best_block = None
        best_overlap = -999999
        for s_abs, e_abs in abs_blocks:
            overlap = max(0, min(e_abs, end_ms) - max(s_abs, start_ms))
            if overlap > best_overlap:
                best_overlap = overlap
                best_block = (s_abs, e_abs)
        if best_block and best_overlap > 0:
            return max(0, best_block[0] - 50), min(len(audio_segment), best_block[1] + 50)
        return start_ms, end_ms
    except Exception:
        return start_ms, end_ms
    finally:
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass

def main():
    # Find all episode directories
    ep_dirs = []
    for name in os.listdir(TTS_DATASET_DIR):
        dir_path = os.path.join(TTS_DATASET_DIR, name)
        if os.path.isdir(dir_path):
            match = re.search(r"_-_(\d{2})$", name)
            if match:
                ep_num = int(match.group(1))
                ep_dirs.append((ep_num, dir_path))
    ep_dirs.sort(key=lambda x: x[0])
    
    print(f"Found {len(ep_dirs)} episodes to scan.")
    
    # DB structures
    db_1sample = {}  # Store only the very first sample per speaker
    db_accum = {}    # Accumulate all samples per speaker (up to 100)
    
    for ep_num, vocals_dir in ep_dirs:
        ep_label = f"EP{ep_num:02d}"
        print(f"\nScanning {ep_label}...")
        
        # Locate vocals wav
        vocals_wav = None
        for filename in os.listdir(vocals_dir):
            if filename.endswith(".wav") and "vocals" in filename.lower():
                vocals_wav = os.path.join(vocals_dir, filename)
                break
        if not vocals_wav:
            continue
            
        # Locate subtitle
        srt_pattern = f"S01E{ep_num:02d}"
        srt_file = None
        for filename in os.listdir(SUBTITLE_DIR):
            if filename.endswith(".srt") and srt_pattern in filename:
                srt_file = os.path.join(SUBTITLE_DIR, filename)
                break
        if not srt_file:
            continue
            
        # Load audio and parse srt
        audio = AudioSegment.from_file(vocals_wav)
        segments = parse_srt(srt_file)
        
        for seg in segments:
            speaker = seg["speaker"]
            # Exclude _unknown or effects
            if speaker == "_unknown" or is_sfx(speaker):
                continue
                
            start_ms = seg["start_ms"]
            end_ms = seg["end_ms"]
            
            # Apply VAD
            ref_start, ref_end = get_local_vad_bounds(audio, start_ms, end_ms)
            dur = (ref_end - ref_start) / 1000.0
            
            # Anchor sample duration constraint
            if dur < 1.5 or dur > 10.0:
                continue
                
            # Extract embedding
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            try:
                clip = audio[ref_start:ref_end]
                clip.set_channels(1).set_frame_rate(16000).export(temp_path, format="wav")
                emb = extract_emb(temp_path)
                if emb is not None:
                    # 1. 1-Sample DB
                    if speaker not in db_1sample:
                        db_1sample[speaker] = [emb]
                        print(f"  [1-Sample DB] Registered first sample for '{speaker}' from {ep_label}")
                    
                    # 2. Accumulated DB
                    if speaker not in db_accum:
                        db_accum[speaker] = []
                    if len(db_accum[speaker]) < 100:
                        db_accum[speaker].append(emb)
                        print(f"  [Accum DB] Added sample for '{speaker}' ({len(db_accum[speaker])}/100)")
            except Exception as e:
                print(f"  Error extracting sample: {e}")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                        
    # Save the databases
    with open(DB_1SAMPLE_PATH, "wb") as f:
        pickle.dump(db_1sample, f)
    print(f"\nSaved 1-sample DB to {DB_1SAMPLE_PATH} (Contains {len(db_1sample)} speakers)")
    
    with open(DB_ACCUM_PATH, "wb") as f:
        pickle.dump(db_accum, f)
    print(f"Saved accumulated DB to {DB_ACCUM_PATH} (Contains {len(db_accum)} speakers)")

if __name__ == "__main__":
    main()
