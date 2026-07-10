#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase A: per-episode extraction (no clustering). Reuses the v6 boundary logic
(CTC align + VAD-snap + hard neighbor-clamp using ALL raw SRT blocks) that
fixed mid-word truncation and cross-speaker bleed. Saves clips + embeddings +
metadata per episode; global speaker identity is assigned later in
identify_global.py so cluster numbering never gets mixed up across episodes.

Usage: python extract_episode.py <ep_id> <srt_path> <vocals_path> <out_dir>
"""
import os, re, sys, csv, warnings, bisect
warnings.filterwarnings("ignore")

EP_ID = sys.argv[1]
SRT = sys.argv[2]
VOCALS = sys.argv[3]
OUTDIR = sys.argv[4]
WESPK_DIR = os.path.expanduser("~/TTSizer/weights/wespeaker-voxceleb-resnet293-LM")
SR = 16000
MIN_DUR, MAX_DUR = 1.0, 12.0

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
        first_paren_pos = t.find("（")
        pre = t[:first_paren_pos].strip() if first_paren_pos >= 0 else t
        if pre == "":
            speaker = parens[0]
        if len(parens) >= 2:
            multi = True
        elif len(parens) == 1 and pre != "" and any(h in parens[0] for h in NONSPEECH_HINTS):
            pass
        elif len(parens) == 1 and pre != "":
            multi = True
    if any(h in raw for h in NONSPEECH_HINTS) and text == "":
        is_nonspeech = True
    if len(parens) == 1 and speaker is None and text == "":
        is_nonspeech = True
    return text, speaker, multi, is_nonspeech

segs = parse_srt(SRT)
print(f"[{EP_ID}] parse: {len(segs)} raw SRT blocks")

kept = []
for s in segs:
    text, speaker, multi, nonsp = clean_and_tag(s["raw"])
    dur = s["end"] - s["start"]
    if nonsp or text == "":
        continue
    if multi:
        continue
    kept.append({**s, "text": text, "speaker": speaker, "dur": dur})
print(f"[{EP_ID}] filter: {len(kept)} candidate speech lines")

if not kept:
    print(f"[{EP_ID}] nothing to do, exiting")
    sys.exit(0)

# ---------- 2. Load audio ----------
wav, sr = sf.read(VOCALS, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
if sr != SR:
    wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    sr = SR
print(f"[{EP_ID}] audio: {len(wav)/sr:.1f}s @ {sr}Hz")

# ---------- 2b. Forced alignment (CTC) ----------
USE_CTC = True
try:
    from ctc_forced_aligner import (load_alignment_model, generate_emissions,
                                     preprocess_text, get_alignments, get_spans,
                                     postprocess_results)
    align_model, align_tok = load_alignment_model(DEVICE, dtype=torch.float32)
    print(f"[{EP_ID}] CTC forced aligner loaded")
except Exception as e:
    USE_CTC = False
    print(f"[{EP_ID}] CTC load failed ({e}); using Silero VAD fallback")

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
    regions = vad_speech_regions(s, e)
    if not regions:
        return s, e
    best = None
    for a, b in regions:
        if a <= e and b >= s:
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
            print(f"[{EP_ID}] CTC probe failed; using Silero VAD fallback")
        else:
            print(f"[{EP_ID}] CTC probe OK")
    else:
        print(f"[{EP_ID}] CTC probe OK")

# ---------- 3. Load wespeaker embedding model ----------
import wespeaker
wmodel = wespeaker.load_model_local(WESPK_DIR)
try:
    wmodel.set_device(DEVICE)
except Exception:
    pass

def embed(a):
    t = torch.from_numpy(a.astype("float32")).unsqueeze(0)
    e = np.asarray(wmodel.extract_embedding_from_pcm(t, SR), dtype="float32")
    return e / (np.linalg.norm(e) + 1e-8)

# All raw SRT boundaries (unfiltered) for the hard neighbor clamp -- this is
# what fixed cross-speaker tail-bleed (v5/v6 fix): clamp against ANY
# neighboring cue, even ones we dropped (e.g. multi-speaker-tagged lines),
# since those still represent real audio belonging to someone else.
ALL_STARTS = sorted(x["start"] for x in segs)
ALL_ENDS = sorted(x["end"] for x in segs)
def next_boundary_after(t):
    i = bisect.bisect_right(ALL_STARTS, t)
    return ALL_STARTS[i] if i < len(ALL_STARTS) else None
def prev_boundary_before(t):
    i = bisect.bisect_left(ALL_ENDS, t)
    return ALL_ENDS[i-1] if i > 0 else None

# ---------- 4. Slice + refine ----------
os.makedirs(OUTDIR, exist_ok=True)
clips = []
ctc_ok = 0; vad_ok = 0; raw_used = 0
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
    s, e = vad_snap(s, e, next_start=next_start)

    # guard pad FIRST, then hard clamp LAST (order bug fix from v5->v6:
    # padding after the clamp was cancelling the clamp's safety margin).
    s = max(0, s-0.05); e = min(len(wav)/sr, e+0.05)
    nb = next_boundary_after(k["start"] + 0.01)
    if nb is not None and e > nb:
        e = nb - 0.05
    pb = prev_boundary_before(k["end"] - 0.01)
    if pb is not None and s < pb:
        s = pb + 0.05

    dur = e - s
    if dur < MIN_DUR or dur > MAX_DUR:
        continue
    clips.append({**k, "cs": s, "ce": e, "cdur": dur})

print(f"[{EP_ID}] slice: {len(clips)} clips (CTC={ctc_ok}, VAD={vad_ok}, raw={raw_used})")

# ---------- 5. Embed + write (no clustering here) ----------
embs = []
meta_rows = []
for c in clips:
    seg = wav[int(c["cs"]*sr):int(c["ce"]*sr)]
    e = embed(seg)
    embs.append(e)
    fname = f"{EP_ID}_{int(c['cs']*1000):08d}_{int(c['ce']*1000):08d}.wav"
    sf.write(os.path.join(OUTDIR, fname), seg, sr)
    meta_rows.append([fname, EP_ID, f"{c['cs']:.3f}", f"{c['ce']:.3f}", f"{c['cdur']:.3f}",
                      c["text"], c["speaker"] or ""])

if embs:
    embs = np.vstack(embs)
    np.savez(os.path.join(OUTDIR, "embeddings.npz"), embs=embs)
    with open(os.path.join(OUTDIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "ep", "start", "end", "dur", "jp_text", "sub_speaker_tag"])
        w.writerows(meta_rows)
print(f"[{EP_ID}] DONE: {len(clips)} clips -> {OUTDIR}")
