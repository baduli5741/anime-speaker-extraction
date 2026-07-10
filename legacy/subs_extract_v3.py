#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Path B v3: Japanese-subtitle-driven speaker extraction (fully local).
Parse JP SRT -> CTC forced-align each line -> VAD-snap boundaries (fix mid-word
truncation) -> reject clips that span a speaker change (intra-clip embedding
consistency) -> embed survivors (wespeaker resnet293) -> over-cluster ->
per-cluster folders + metadata.csv
"""
import os, re, sys, csv, glob, json, math, warnings
warnings.filterwarnings("ignore")

SRT = os.path.expanduser("~/frieren_subs/ep01.srt")
VOCALS = os.path.expanduser("~/ttsizer_out/Frieren/vocals_normalized/frieren_ep01_vocals.flac")
OUTDIR = os.path.expanduser("~/subs_out_v3/ep01")
WESPK_DIR = os.path.expanduser("~/TTSizer/weights/wespeaker-voxceleb-resnet293-LM")
SR = 16000
MIN_DUR, MAX_DUR = 1.0, 12.0
MULTI_SPK_COS_THRESH = float(os.environ.get("MULTI_SPK_THRESH", "0.55"))  # below = likely 2 speakers
CLUSTER_THRESH = float(os.environ.get("CLUSTER_THRESH", "0.45"))

import numpy as np
import torch, torchaudio
import soundfile as sf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Parse SRT ----------
TAG_RE = re.compile(r"\{\\[^}]*\}")
PAREN_RE = re.compile(r"（([^）]*)）")
NONSPEECH_HINTS = ("音", "笑い声", "ざわめき", "鳴き", "効果音", "拍手", "足音", "声）")

def ts(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def parse_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    blocks = re.split(r"\r?\n\r?\n", raw.strip())
    out = []
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip() != ""]
        if len(lines) < 2:
            continue
        tl = next((l for l in lines if "-->" in l), None)
        if tl is None:
            continue
        idx = lines.index(tl)
        start, end = [x.strip() for x in tl.split("-->")]
        text_lines = lines[idx+1:]
        out.append({"start": ts(start), "end": ts(end), "raw": " ".join(text_lines)})
    return out

def clean_and_tag(raw):
    t = TAG_RE.sub("", raw).strip()
    speaker = None
    multi = False
    parens = PAREN_RE.findall(t)
    text = PAREN_RE.sub("", t).strip()
    is_nonspeech = False
    if len(parens) >= 1:
        # first paren before any real text = speaker tag; any paren embedded
        # mid-line after real text already appeared = a 2nd speaker cut in
        first_paren_pos = t.find("（")
        pre = t[:first_paren_pos].strip() if first_paren_pos >= 0 else t
        if len(parens) >= 1 and pre == "":
            speaker = parens[0]
        if len(parens) >= 2:
            multi = True  # more than one speaker tag on this line -> drop
        elif len(parens) == 1 and pre != "" and PAREN_RE.search(t) and \
                any(h in parens[0] for h in NONSPEECH_HINTS):
            pass  # trailing sfx note, ignore
        elif len(parens) == 1 and pre != "":
            multi = True  # text then (speaker) again mid-line -> 2nd speaker
    if any(h in raw for h in NONSPEECH_HINTS) and text == "":
        is_nonspeech = True
    if len(parens) == 1 and speaker is None and text == "":
        is_nonspeech = True
    return text, speaker, multi, is_nonspeech

segs = parse_srt(SRT)
print(f"[parse] {len(segs)} raw SRT blocks")

kept = []
for s in segs:
    text, speaker, multi, nonsp = clean_and_tag(s["raw"])
    dur = s["end"] - s["start"]
    if nonsp or text == "":
        continue
    if multi:
        continue
    kept.append({**s, "text": text, "speaker": speaker, "dur": dur})
print(f"[filter] {len(kept)} candidate speech lines (after nonspeech/multi drop)")

# ---------- 2. Load audio ----------
wav, sr = sf.read(VOCALS, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
if sr != SR:
    wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    sr = SR
print(f"[audio] {len(wav)/sr:.1f}s @ {sr}Hz")

# ---------- 2b. Forced alignment (CTC) ----------
USE_CTC = True
try:
    from ctc_forced_aligner import (load_alignment_model, generate_emissions,
                                     preprocess_text, get_alignments, get_spans,
                                     postprocess_results)
    align_model, align_tok = load_alignment_model(DEVICE, dtype=torch.float32)
    print("[align] CTC forced aligner loaded")
except Exception as e:
    USE_CTC = False
    print(f"[align] CTC load failed ({e}); using Silero VAD fallback")

# Silero VAD
vad_model, vad_utils = torch.hub.load(
    repo_or_dir=os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master"),
    model="silero_vad", source="local", trust_repo=True) \
    if os.path.isdir(os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master")) else (None, None)
if vad_model is None:
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad_model = load_silero_vad()
    _get_ts = get_speech_timestamps
else:
    (_get_ts, _, _, _, _) = vad_utils
vad_model.to("cpu")

def vad_speech_regions(a_start, a_end, pad_pre=0.1, pad_post=0.6):
    i0 = max(0, int((a_start-pad_pre)*sr)); i1 = min(len(wav), int((a_end+pad_post)*sr))
    if i1 - i0 < int(0.2*sr):
        return []
    chunk = torch.from_numpy(wav[i0:i1]).float()
    tss = _get_ts(chunk, vad_model, sampling_rate=sr, threshold=0.35)
    return [(i0/sr + t["start"]/sr, i0/sr + t["end"]/sr) for t in tss]

def vad_snap(s, e, next_start=None):
    """Extend (s,e) to the full VAD speech region overlapping it, so we don't
    cut off mid-word/mid-phrase. Capped extension; never crosses into the next
    subtitle line's start (to avoid grabbing the next speaker)."""
    regions = vad_speech_regions(s, e)
    if not regions:
        return s, e
    best = None
    for a, b in regions:
        if a <= e and b >= s:  # overlaps the CTC/subtitle window
            if best is None or (b - a) > (best[1] - best[0]):
                best = (a, b)
    if best is None:
        return s, e
    new_s = max(s - 0.3, min(s, best[0]))
    new_e = min(e + 0.6, max(e, best[1]))
    if next_start is not None:
        new_e = min(new_e, next_start - 0.05)
    if new_e <= new_s:
        return s, e
    return new_s, new_e

PAD = 0.35
def ctc_refine(a_start, a_end, text):
    i0 = max(0, int((a_start-PAD)*sr)); i1 = min(len(wav), int((a_end+PAD)*sr))
    chunk = wav[i0:i1]
    if len(chunk) < int(0.2*sr):
        return None
    try:
        wt = torch.from_numpy(np.ascontiguousarray(chunk)).float()
        emissions, stride = generate_emissions(align_model, wt.to(DEVICE), batch_size=1)
        tok_starred, text_starred = preprocess_text(text, romanize=True, language="jpn")
        segments, scores, blank = get_alignments(emissions, tok_starred, align_tok)
        spans = get_spans(tok_starred, segments, blank)
        results = postprocess_results(text_starred, spans, stride, scores)
        if not results:
            return None
        s = (i0/sr) + min(r["start"] for r in results)
        e = (i0/sr) + max(r["end"] for r in results)
        return s, e
    except Exception:
        return None

if USE_CTC:
    ps, pe = kept[0]["start"], kept[0]["end"]
    if ctc_refine(ps, pe, kept[0]["text"]) is None:
        if all(ctc_refine(kept[j]["start"], kept[j]["end"], kept[j]["text"]) is None
               for j in range(1, min(4, len(kept)))):
            USE_CTC = False
            print("[align] CTC probe failed (likely uroman/perl); using Silero VAD fallback")
        else:
            print("[align] CTC probe OK")
    else:
        print("[align] CTC probe OK")

# ---------- 3. Load embedding model BEFORE slicing (needed for multi-speaker check) ----------
import wespeaker
wmodel = wespeaker.load_model_local(WESPK_DIR)
try:
    wmodel.set_device(DEVICE)
except Exception:
    pass

def embed(a):
    t = torch.from_numpy(a.astype("float32")).unsqueeze(0)
    emb = wmodel.extract_embedding_from_pcm(t, SR)
    e = np.asarray(emb, dtype="float32")
    return e / (np.linalg.norm(e) + 1e-8)

def is_multi_speaker(seg):
    """Split the clip into an early and late window (skipping the middle) and
    compare embeddings. Low similarity => the clip likely spans a speaker
    change (e.g. one subtitle cue capturing two people back-to-back)."""
    n = len(seg)
    if n < int(1.3*sr):
        return False, 1.0  # too short to split reliably; don't reject on this basis
    a = seg[:int(0.42*n)]
    b = seg[int(0.58*n):]
    if len(a) < int(0.35*sr) or len(b) < int(0.35*sr):
        return False, 1.0
    try:
        ea, eb = embed(a), embed(b)
        cos = float(np.dot(ea, eb))
    except Exception:
        return False, 1.0
    return (cos < MULTI_SPK_COS_THRESH), cos

# ---------- 4. Slice + refine + multi-speaker filter ----------
os.makedirs(OUTDIR, exist_ok=True)
clips = []
ctc_ok = 0; vad_ok = 0; raw_used = 0; snapped = 0; dropped_multi = 0; dropped_len = 0
for i, k in enumerate(kept):
    bounds = None
    if USE_CTC:
        bounds = ctc_refine(k["start"], k["end"], k["text"])
        if bounds: ctc_ok += 1
    if bounds is None:
        regions = vad_speech_regions(k["start"], k["end"], pad_pre=0.05, pad_post=0.05)
        if regions:
            bounds = (min(r[0] for r in regions), max(r[1] for r in regions)); vad_ok += 1
    if bounds is None:
        bounds = (k["start"], k["end"]); raw_used += 1
    s, e = bounds

    next_start = kept[i+1]["start"] if i+1 < len(kept) else None
    s2, e2 = vad_snap(s, e, next_start=next_start)
    if (s2, e2) != (s, e):
        snapped += 1
    s, e = s2, e2

    s = max(0, s-0.05); e = min(len(wav)/sr, e+0.05)
    dur = e - s
    if dur < MIN_DUR or dur > MAX_DUR:
        dropped_len += 1
        continue

    seg = wav[int(s*sr):int(e*sr)]
    is_multi, cos = is_multi_speaker(seg)
    if is_multi:
        dropped_multi += 1
        continue

    clips.append({**k, "cs": s, "ce": e, "cdur": dur, "intra_cos": cos})

print(f"[slice] {len(clips)} clips kept (CTC={ctc_ok}, VAD={vad_ok}, raw={raw_used}, "
      f"vad_snapped={snapped}, dropped_multi_speaker={dropped_multi}, dropped_len={dropped_len})")

# ---------- 5. Embeddings for clustering ----------
embs = []
for c in clips:
    seg = wav[int(c["cs"]*sr):int(c["ce"]*sr)]
    embs.append(embed(seg))
embs = np.vstack(embs)
print(f"[embed] {embs.shape}")
np.savez(os.path.join(OUTDIR, "embeddings.npz"), embs=embs,
         cs=np.array([c["cs"] for c in clips]), ce=np.array([c["ce"] for c in clips]))

# ---------- 6. Over-cluster ----------
from sklearn.cluster import AgglomerativeClustering
labels = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_THRESH,
                                 metric="cosine", linkage="average").fit_predict(embs)
nclust = len(set(labels))
print(f"[cluster] {nclust} clusters")

# ---------- 7. Write outputs ----------
for lb in set(labels):
    os.makedirs(os.path.join(OUTDIR, f"cluster_{lb:02d}"), exist_ok=True)

meta_path = os.path.join(OUTDIR, "metadata.csv")
with open(meta_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["clip","cluster","start","end","dur","jp_text","sub_speaker_tag","intra_cos"])
    for c, lb in zip(clips, labels):
        cid = f"cluster_{lb:02d}"
        fname = f"{cid}/ep01_{int(c['cs']*1000):08d}_{int(c['ce']*1000):08d}.wav"
        seg = wav[int(c["cs"]*sr):int(c["ce"]*sr)]
        sf.write(os.path.join(OUTDIR, fname), seg, sr)
        w.writerow([fname, cid, f"{c['cs']:.3f}", f"{c['ce']:.3f}",
                    f"{c['cdur']:.3f}", c["text"], c["speaker"] or "", f"{c['intra_cos']:.3f}"])

# ---------- summary ----------
from collections import Counter
cc = Counter(labels)
durs = np.array([c["cdur"] for c in clips])
print("=== SUMMARY ===")
print(f"clips={len(clips)} clusters={nclust}")
print(f"dur min/med/max = {durs.min():.2f}/{np.median(durs):.2f}/{durs.max():.2f}s")
print("clips per cluster:", dict(sorted(cc.items())))
tagmap = {}
for c, lb in zip(clips, labels):
    if c["speaker"]:
        tagmap.setdefault(lb, Counter())[c["speaker"]] += 1
print("cluster -> subtitle tag hints:")
for lb in sorted(tagmap):
    print(f"  cluster_{lb:02d}: {dict(tagmap[lb].most_common(3))}")
print("METADATA:", meta_path)
