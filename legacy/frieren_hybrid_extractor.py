import os
import sys
import io
import re
import shutil
import tempfile
import torch
import torchaudio
from pydub import AudioSegment
import imageio_ffmpeg
import numpy as np

# Force UTF-8 Output on Windows system consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure ffmpeg path dynamically
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

# -------------------------------------------------------------
# 1. Silero VAD Loading
# -------------------------------------------------------------
print("Loading Silero VAD Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
vad_model = vad_model.to(device)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
print(f"Silero VAD loaded on device: {device}")

# -------------------------------------------------------------
# 2. SFX Filter Keywords
# -------------------------------------------------------------
SFX_KEYWORDS = [
    "笑い声", "歓声", "拍手", "足音", "進行音", "爆発", "効果音",
    "鳴き声", "ドア", "鐘", "風", "雨", "音楽", "♪", "～♪",
    "叫び声", "悲鳴", "銃声", "物音", "ため息", "咳", "うなり",
    "ノック", "電話", "サイレン", "轟音", "地響き", "水音",
]

def is_sfx(text):
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
            return None, None  # Multiple speakers
        if re.search(r'\d+人', candidate_name):
            return None, None
            
        speaker = candidate_name
        text = remaining if remaining else None
    
    if text:
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'<[^>]*>', '', text)
        text = text.replace('\r', '').replace('\n', ' ').strip()
        
    return speaker, text

# -------------------------------------------------------------
# 3. Parse SRT with Speaker States and Temporal Merging
# -------------------------------------------------------------
def parse_srt(srt_path, state_gap_ms=500, merge_gap_ms=800, max_clip_sec=12.0, merge_consecutive=True):
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    if not blocks or (len(blocks) == 1 and not blocks[0].strip()):
        blocks = content.strip().split('\r\n\r\n')

    segments = []
    current_speaker = None
    last_end_ms = 0

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
                
        if not time_line:
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
        if text is None and speaker is None:
            continue

        # Speaker context carrying
        if speaker:
            current_speaker = speaker
        else:
            gap = start_ms - last_end_ms
            if gap > state_gap_ms:
                current_speaker = None

        if not text:
            continue

        last_end_ms = end_ms
        segments.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speaker": current_speaker,
            "text": text
        })

    if not merge_consecutive:
        return segments

    if not segments:
        return []

    merged = []
    curr = segments[0].copy()

    for next_seg in segments[1:]:
        gap = next_seg["start_ms"] - curr["end_ms"]
        total_dur = (next_seg["end_ms"] - curr["start_ms"]) / 1000.0

        if (curr["speaker"] == next_seg["speaker"] and
            curr["speaker"] is not None and
            gap >= 0 and gap <= merge_gap_ms and
            total_dur <= max_clip_sec):
            
            curr["end_ms"] = next_seg["end_ms"]
            curr["text"] = curr["text"] + " " + next_seg["text"]
        else:
            merged.append(curr)
            curr = next_seg.copy()

    merged.append(curr)
    return merged

# -------------------------------------------------------------
# 4. Precision Slicing using Local Silero VAD Slicing
# -------------------------------------------------------------
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
            ref_start = max(0, best_block[0] - 50)
            ref_end = min(len(audio_segment), best_block[1] + 50)
            return ref_start, ref_end
        else:
            return start_ms, end_ms
    except Exception as e:
        print(f"VAD Exception: {e}")
        return start_ms, end_ms
    finally:
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass

# -------------------------------------------------------------
# 5. Core Processing Pipeline
# -------------------------------------------------------------
def process_episode(vocals_wav_path, srt_path, output_dir, episode_label):
    print(f"\nProcessing episode {episode_label}...")
    print(f"Vocals WAV: {vocals_wav_path}")
    print(f"Subtitle: {srt_path}")
    
    # Check if target subfolders exist and delete them for clean process
    for filename in os.listdir(output_dir):
        dir_path = os.path.join(output_dir, filename)
        if os.path.isdir(dir_path):
            if filename.startswith(f"EP") or filename == "_unknown":
                try:
                    shutil.rmtree(dir_path)
                except Exception:
                    pass

    # Load vocals track
    print("Loading audio track...")
    audio = AudioSegment.from_file(vocals_wav_path)
    print(f"Audio Loaded. Duration: {len(audio)/1000:.1f}s")
    
    # Parse subtitles without pre-merging to capture individual blocks
    segments = parse_srt(srt_path, state_gap_ms=0, merge_consecutive=False)
    print(f"Parsed into {len(segments)} raw target segments.")
    if not segments:
        print("No segments found in subtitle.")
        return
        
    # Load SpeechBrain Classifier
    print("Initializing SpeechBrain for Multi-Speaker Verification...")
    from speechbrain.inference.speaker import EncoderClassifier
    
    sb_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    def extract_emb(path):
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
        return emb

    # =============================================================
    # PASS 1: Rough Slicing and Extracting Embeddings
    # =============================================================
    print("Executing Pass 1: Extracting raw speaker slices and embeddings via local VAD...")
    pre_slices = []
    
    for idx, seg in enumerate(segments):
        speaker = seg["speaker"] or "_unknown"
        text = seg["text"]
        start_ms = seg["start_ms"]
        end_ms = seg["end_ms"]
        
        ref_start, ref_end = get_local_vad_bounds(audio, start_ms, end_ms)
        
        duration = (ref_end - ref_start) / 1000.0
        if duration < 0.5:
            continue
            
        pre_slices.append({
            "idx": idx,
            "text": text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "ref_start": ref_start,
            "ref_end": ref_end,
            "raw_speaker": speaker
        })

    # Deduplicate and merge overlapping physical audio bounds (Overlap > 0 or Gap <= 100ms)
    merged_pre_slices = []
    for s in pre_slices:
        if not merged_pre_slices:
            merged_pre_slices.append(s)
        else:
            prev = merged_pre_slices[-1]
            overlap = prev["ref_end"] - s["ref_start"]
            gap = s["ref_start"] - prev["ref_end"]
            
            can_merge_speaker = (prev["raw_speaker"] == s["raw_speaker"]) or (prev["raw_speaker"] == "_unknown" or s["raw_speaker"] == "_unknown")
            potential_dur = (max(prev["ref_end"], s["ref_end"]) - min(prev["ref_start"], s["ref_start"])) / 1000.0
            
            if can_merge_speaker and (overlap > 0 or gap <= 100) and potential_dur <= 15.0:
                print(f"  [Pass 1 Overlap Merge] Merging slice {prev['idx']} and {s['idx']} due to physical overlap ({gap}ms gap).")
                prev["ref_start"] = min(prev["ref_start"], s["ref_start"])
                prev["ref_end"] = max(prev["ref_end"], s["ref_end"])
                prev["text"] = prev["text"] + " " + s["text"]
                if prev["raw_speaker"] == "_unknown" and s["raw_speaker"] != "_unknown":
                    prev["raw_speaker"] = s["raw_speaker"]
            else:
                merged_pre_slices.append(s)

    raw_slices = []
    for s in merged_pre_slices:
        duration = (s["ref_end"] - s["ref_start"]) / 1000.0
        if duration < 1.0 or duration > 15.0:
            continue
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            clip = audio[s["ref_start"]:s["ref_end"]]
            clip.set_channels(1).set_frame_rate(22050).export(temp_path, format="wav")
            emb = extract_emb(temp_path)
            norm = np.linalg.norm(emb)
            
            s["temp_path"] = temp_path
            s["duration"] = duration
            s["emb"] = emb / norm if norm > 0 else None
            raw_slices.append(s)
        except Exception as e:
            print(f"Error in Pass 1 for slice {s['idx']}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # =============================================================
    # PASS 2: Compute Self-Bootstrapped Centroids Dynamically
    # =============================================================
    print("Executing Pass 2: Computing self-bootstrapped centroids...")
    import pickle
    db_path = os.path.join(output_dir, "speaker_embeddings_db.pkl")
    
    spk_db = {}
    if os.path.exists(db_path):
        try:
            with open(db_path, "rb") as f:
                spk_db = pickle.load(f)
            print(f"Loaded existing speaker embeddings database from {db_path}.")
            for spk, embs in spk_db.items():
                print(f"  '{spk}': {len(embs)} samples currently in DB.")
        except Exception as e:
            print(f"Warning: Failed to load embeddings database: {e}. Starting fresh.")
            spk_db = {}

    # Extract ALL unique raw speaker names dynamically from raw slices (except _unknown)
    detected_speakers = set([s["raw_speaker"] for s in raw_slices if s["raw_speaker"] != "_unknown"])
    print(f"Dynamically detected speakers in this episode: {list(detected_speakers)}")
    
    centroids = {}
    for spk in detected_speakers:
        if spk not in spk_db:
            spk_db[spk] = []
            
        candidates = [s for s in raw_slices if s["raw_speaker"] == spk and s["emb"] is not None and s["duration"] >= 1.5]
        if candidates:
            new_embs = [s["emb"] for s in candidates]
            spk_db[spk].extend(new_embs)
            if len(spk_db[spk]) > 100:
                spk_db[spk] = spk_db[spk][-100:]
            print(f"  Added {len(candidates)} new native samples for speaker '{spk}'. Total DB size: {len(spk_db[spk])}.")

    # Compute centroids for all accumulated speakers in the DB (need >= 1 samples)
    for spk, embs in spk_db.items():
        if len(embs) >= 1:
            mean_emb = np.mean(embs, axis=0)
            centroids[spk] = mean_emb / np.linalg.norm(mean_emb)
            print(f"  Computed bootstrapped centroid for '{spk}' using {len(embs)} samples.")

    # Save updated database
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(db_path, "wb") as f:
            pickle.dump(spk_db, f)
        print(f"Saved speaker embeddings database to {db_path}.")
    except Exception as e:
        print(f"Error saving speaker database: {e}")

    # =============================================================
    # PASS 3: Dynamic Routing & Post-Verification Merging
    # =============================================================
    print("Executing Pass 3: Routing and post-verification merging...")
    
    for s in raw_slices:
        speaker = s["raw_speaker"]
        text = s["text"]
        emb = s["emb"]
        
        verified_speaker = "_unknown"
        if emb is not None and len(centroids) > 0:
            sims = {}
            for ch, cent in centroids.items():
                sims[ch] = float(np.dot(emb, cent))
                
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            best_spk, best_sim = sorted_sims[0]
            second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else -1.0
            
            # Routing Decision
            if speaker != "_unknown":
                expected = speaker
                if expected in centroids:
                    exp_sim = sims[expected]
                    if exp_sim >= 0.50:
                        verified_speaker = expected
                    else:
                        if best_sim >= 0.40 and (best_sim - second_sim) >= 0.15:
                            print(f"  [Re-route/Margin] {expected} slice '{text[:15]}...' expected_sim={exp_sim:.4f} < 0.50. But matched {best_spk} via Margin (sim={best_sim:.4f}, margin={best_sim - second_sim:.4f}).")
                            verified_speaker = best_spk
                        else:
                            verified_speaker = "_unknown"
                else:
                    if best_sim >= 0.40 and (best_sim - second_sim) >= 0.15:
                        verified_speaker = best_spk
                    else:
                        verified_speaker = speaker
            else:
                if best_sim >= 0.70:
                    verified_speaker = best_spk
                elif best_sim >= 0.40 and (best_sim - second_sim) >= 0.15:
                    verified_speaker = best_spk
                else:
                    verified_speaker = "_unknown"
        else:
            verified_speaker = speaker
            
        s["verified_speaker"] = verified_speaker

    # 3.2: Group adjacent slices of the SAME verified speaker if gap <= 800ms
    merged_groups = []
    for s in raw_slices:
        if not merged_groups:
            merged_groups.append([s])
        else:
            prev_group = merged_groups[-1]
            prev_s = prev_group[-1]
            gap = s["ref_start"] - prev_s["ref_end"]
            potential_dur = (s["ref_end"] - prev_group[0]["ref_start"]) / 1000.0
            
            if (s["verified_speaker"] == prev_s["verified_speaker"] and
                s["verified_speaker"] != "_unknown" and
                gap >= 0 and gap <= 800 and
                potential_dur <= 12.0):
                prev_group.append(s)
            else:
                merged_groups.append([s])

    # 3.3: Crop, save final WAV files and record statistics
    stats = {}
    for group in merged_groups:
        verified_speaker = group[0]["verified_speaker"]
        ref_start = group[0]["ref_start"]
        ref_end = group[-1]["ref_end"]
        text = " ".join([s["text"] for s in group])
        duration = (ref_end - ref_start) / 1000.0
        
        if len(group) > 1:
            print(f"  [Post-Merge] Merged {len(group)} slices for '{verified_speaker}': '{text[:25]}...' ({duration:.2f}s)")
            
        if verified_speaker not in stats:
            stats[verified_speaker] = {"count": 0, "metadata": []}
            
        # Create directory exactly using original speaker name (Kanji/Katakana)
        spk_dir = os.path.join(output_dir, verified_speaker)
        os.makedirs(spk_dir, exist_ok=True)
        
        orig_idx = group[0]["idx"]
        filename = f"{verified_speaker}_{episode_label}_idx{orig_idx}.wav"
        file_path = os.path.join(spk_dir, filename)
        
        try:
            clip = audio[ref_start:ref_end]
            clip.set_channels(1).set_frame_rate(22050).export(file_path, format="wav")
            print(f"  [{verified_speaker}] Saved -> {filename} ({duration:.2f}s)")
        except Exception as e:
            print(f"  Error exporting clip {filename}: {e}")
            
        stats[verified_speaker]["count"] += 1
        stats[verified_speaker]["metadata"].append(f"{filename}|{text}|JA")

    # Clean up temp WAV files
    for s in raw_slices:
        temp_path = s["temp_path"]
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    # Write metadata.txt inside each folder
    for speaker, spk_data in stats.items():
        if spk_data["count"] > 0:
            spk_dir = os.path.join(output_dir, speaker)
            metadata_file = os.path.join(spk_dir, "metadata.txt")
            with open(metadata_file, "w", encoding="utf-8") as f:
                f.write("\n".join(spk_data["metadata"]) + "\n")
            print(f"Metadata file written for speaker '{speaker}': {spk_data['count']} clips.")

if __name__ == "__main__":
    if len(sys.argv) >= 5:
        vocals_path = sys.argv[1]
        srt_path = sys.argv[2]
        out_dir = sys.argv[3]
        ep_label = sys.argv[4]
    else:
        vocals_path = r"D:\anime\tts_dataset\[EMBER]_Sousou_no_Frieren_-_01\[EMBER] Sousou no Frieren - 01_vocals_ensemble.wav"
        srt_path = r"C:\Users\piai\Desktop\dialect_and_survey_backup\frieren_sub_netflix\Frieren_.Beyond.Journey's.End.S01E01.WEBRip.Netflix.ja[cc].srt"
        out_dir = r"D:\anime\extracted_dataset_test"
        ep_label = "EP01"
        
    process_episode(vocals_path, srt_path, out_dir, ep_label)
