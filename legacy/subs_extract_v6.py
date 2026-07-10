#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Path B: Japanese-subtitle-driven speaker extraction (fully local).
Parse JP SRT -> forced-align each line to vocals -> slice tight clips ->
embed (wespeaker resnet293) -> over-cluster -> per-cluster folders + metadata.csv
"""
import os, re, sys, csv, glob, json, math, warnings
warnings.filterwarnings("ignore")

SRT = os.path.expanduser("~/frieren_subs/ep01.srt")
VOCALS = os.path.expanduser("~/ttsizer_out/Frieren/vocals_normalized/frieren_ep01_vocals.flac")
OUTDIR = os.path.expanduser("~/subs_out_v6/ep01")
WESPK_DIR = os.path.expanduser("~/TTSizer/weights/wespeaker-voxceleb-resnet293-LM")
SR = 16000
MIN_DUR, MAX_DUR = 1.0, 12.0

import numpy as np
import torch, torchaudio
import soundfile as sf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Parse SRT ----------
TAG_RE = re.compile(r"\{\\[^}]*\}")            # {\an8} etc
PAREN_RE = re.compile(r"（([^）]*)）")          # full-width paren groups
# non-speech cues we always drop if the WHOLE line is one
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
        # find timing line
        tl = next((l for l in lines if "-->" in l), None)
        if tl is None:
            continue
        idx = lines.index(tl)
        start, end = [x.strip() for x in tl.split("-->")]
        text_lines = lines[idx+1:]
        text = " ".join(text_lines)
        text = TAG_RE.sub("", text).strip()
        out.append({"start": ts(start), "end": ts(end), "raw": text})
    return out

def clean_and_tag(raw):
    """Return (clean_text, speaker_tag, multi_speaker_flag, is_nonspeech)."""
    parens = PAREN_RE.findall(raw)
    speaker = None
    multi = False
    # leading tag = speaker; count of paren groups > tells multi-speaker / cue
    stripped = raw
    if parens:
        # if first token is a paren group at very start -> speaker name candidate
        m = re.match(r"^\s*（([^）]*)）", raw)
        if m:
            cand = m.group(1)
            # if it looks like a sound cue, treat as nonspeech marker, not speaker
            if any(h in cand for h in NONSPEECH_HINTS):
                pass
            else:
                speaker = cand
        # remove ALL paren groups from text
    text = PAREN_RE.sub("", raw)
    text = re.sub(r"[♪～\s]+", "", text) if re.sub(r"[♪～\s]", "", raw) == "" else text
    text = text.replace("♪", "").replace("～", "")
    # remove ruby readings like 凱旋(がいせん) -> keep base, drop paren reading (half-width)
    text = re.sub(r"\(([^)]*)\)", "", text)
    text = text.strip()
    # multi-speaker: a second（name）tag embedded after some text
    non_cue_tags = [p for p in parens if not any(h in p for h in NONSPEECH_HINTS)]
    if len(non_cue_tags) >= 2:
        multi = True
    # nonspeech if empty text after cleaning, OR whole raw was a single cue
    is_nonspeech = (text == "")
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
        continue                      # drop multi-speaker lines
    kept.append({**s, "text": text, "speaker": speaker, "dur": dur})
print(f"[filter] {len(kept)} candidate speech lines (after nonspeech/multi drop)")

# All raw SRT block boundaries (unfiltered) -- used to clamp against ANY
# neighboring line (even ones we dropped, e.g. multi-speaker-tagged), since
# those still represent real audio from another speaker.
ALL_STARTS = sorted(x["start"] for x in segs)
ALL_ENDS = sorted(x["end"] for x in segs)
import bisect
def next_boundary_after(t):
    i = bisect.bisect_right(ALL_STARTS, t)
    return ALL_STARTS[i] if i < len(ALL_STARTS) else None
def prev_boundary_before(t):
    i = bisect.bisect_left(ALL_ENDS, t)
    return ALL_ENDS[i-1] if i > 0 else None

# ---------- 2. Load audio ----------
wav, sr = sf.read(VOCALS, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
if sr != SR:
    wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    sr = SR
print(f"[audio] {len(wav)/sr:.1f}s @ {sr}Hz")

# ---------- 2b. Forced alignment (CTC) with VAD fallback ----------
USE_CTC = True
try:
    from ctc_forced_aligner import (load_alignment_model, generate_emissions,
                                     preprocess_text, get_alignments, get_spans,
                                     postprocess_results)
    # NOTE: keep float32 — fp16 causes "Input type (float) and bias type (Half)" in wav2vec2 conv
    align_model, align_tok = load_alignment_model(DEVICE, dtype=torch.float32)
    print("[align] CTC forced aligner loaded")
except Exception as e:
    USE_CTC = False
    print(f"[align] CTC load failed ({e}); using Silero VAD fallback")

# Silero VAD (always load for fallback / trimming)
vad_model, vad_utils = torch.hub.load(repo_or_dir=os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master"),
                                      model="silero_vad", source="local", trust_repo=True) \
    if os.path.isdir(os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master")) else (None, None)
if vad_model is None:
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad_model = load_silero_vad()
    _get_ts = get_speech_timestamps
else:
    (_get_ts, _, _, _, _) = vad_utils
vad_model.to("cpu")

def vad_refine(a_start, a_end):
    """Trim to speech region within [a_start,a_end] using Silero VAD. Returns (s,e) or None."""
    i0, i1 = int(a_start*sr), int(a_end*sr)
    chunk = torch.from_numpy(wav[i0:i1]).float()
    if len(chunk) < int(0.2*sr):
        return None
    tss = _get_ts(chunk, vad_model, sampling_rate=sr, threshold=0.4)
    if not tss:
        return None
    s = a_start + tss[0]["start"]/sr
    e = a_start + tss[-1]["end"]/sr
    return s, e

# CTC alignment per line, constrained to the subtitle window (+pad)
PAD = 0.35
def ctc_refine(a_start, a_end, text):
    i0 = max(0, int((a_start-PAD)*sr)); i1 = min(len(wav), int((a_end+PAD)*sr))
    chunk = wav[i0:i1]
    if len(chunk) < int(0.2*sr):
        return None
    try:
        # generate_emissions expects a 1D [samples] tensor (it adds batch dim internally)
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

# Probe CTC once (Japanese uroman/perl often broken on compute nodes). If it
# fails, disable CTC up front instead of failing 324x, and use VAD fallback.
if USE_CTC:
    ps, pe = kept[0]["start"], kept[0]["end"]
    if ctc_refine(ps, pe, kept[0]["text"]) is None:
        # try a couple more before giving up (first line could be atypical)
        if all(ctc_refine(kept[j]["start"], kept[j]["end"], kept[j]["text"]) is None
               for j in range(1, min(4, len(kept)))):
            USE_CTC = False
            print("[align] CTC probe failed (likely uroman/perl); using Silero VAD fallback")
        else:
            print("[align] CTC probe OK")
    else:
        print("[align] CTC probe OK")

# ---------- 3. Slice + refine ----------
os.makedirs(OUTDIR, exist_ok=True)
clips = []
ctc_ok = 0; vad_ok = 0; raw_used = 0
for i, k in enumerate(kept):
    bounds = None
    if USE_CTC:
        bounds = ctc_refine(k["start"], k["end"], k["text"])
        if bounds: ctc_ok += 1
    if bounds is None:
        bounds = vad_refine(k["start"], k["end"])
        if bounds: vad_ok += 1
    if bounds is None:
        bounds = (k["start"], k["end"]); raw_used += 1
    s, e = bounds
    # HARD SAFETY CLAMP: never let a clip cross into the next subtitle line's
    # start (this is what caused Eisen+Frieren bleed: fast back-to-back
    # dialogue where one cue's nominal end overlaps the next cue's start).
    # guard pad FIRST, then clamp LAST so the clamp is authoritative and
    # can't be undone by the pad (previous bug: pad re-added after clamp
    # let clips creep back into the next speaker's onset).
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

print(f"[slice] {len(clips)} clips kept (CTC={ctc_ok}, VAD={vad_ok}, raw={raw_used})")

# ---------- 4. Embeddings (wespeaker resnet293) ----------
import wespeaker
wmodel = wespeaker.load_model_local(WESPK_DIR)
try:
    wmodel.set_device(DEVICE)
except Exception:
    pass

def embed(a):
    t = torch.from_numpy(a.astype("float32")).unsqueeze(0)
    # wespeaker expects 16k mono tensor
    emb = wmodel.extract_embedding_from_pcm(t, SR)
    return np.asarray(emb, dtype="float32")

embs = []
tmpwav = os.path.join(OUTDIR, "_tmp.wav")
for c in clips:
    seg = wav[int(c["cs"]*sr):int(c["ce"]*sr)]
    embs.append(embed(seg))
embs = np.vstack(embs)
# L2 normalize
embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
print(f"[embed] {embs.shape}")
# cache embeddings + clip meta so re-clustering needs no GPU
np.savez(os.path.join(OUTDIR, "embeddings.npz"), embs=embs,
         cs=np.array([c["cs"] for c in clips]), ce=np.array([c["ce"] for c in clips]))

# ---------- 5. Over-cluster ----------
from sklearn.cluster import AgglomerativeClustering
# over-cluster: distance threshold on cosine, no forced n_clusters.
# 0.45 chosen empirically: 0.40 -> ~100 clusters (too many singletons),
# 0.50 -> ~58 (one blob mixes speakers); 0.45 balances granularity vs usability.
THRESH = float(os.environ.get("CLUSTER_THRESH", "0.45"))
labels = AgglomerativeClustering(n_clusters=None, distance_threshold=THRESH,
                                 metric="cosine", linkage="average").fit_predict(embs)
nclust = len(set(labels))
print(f"[cluster] {nclust} clusters")

# ---------- 6. Write outputs ----------
for lb in set(labels):
    os.makedirs(os.path.join(OUTDIR, f"cluster_{lb:02d}"), exist_ok=True)

meta_path = os.path.join(OUTDIR, "metadata.csv")
with open(meta_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["clip","cluster","start","end","dur","jp_text","sub_speaker_tag"])
    for c, lb in zip(clips, labels):
        cid = f"cluster_{lb:02d}"
        # include start+end ms so clips starting in the same ms don't collide
        fname = f"{cid}/ep01_{int(c['cs']*1000):08d}_{int(c['ce']*1000):08d}.wav"
        seg = wav[int(c["cs"]*sr):int(c["ce"]*sr)]
        sf.write(os.path.join(OUTDIR, fname), seg, sr)
        w.writerow([fname, cid, f"{c['cs']:.3f}", f"{c['ce']:.3f}",
                    f"{c['cdur']:.3f}", c["text"], c["speaker"] or ""])
if os.path.exists(tmpwav): os.remove(tmpwav)

# ---------- summary ----------
from collections import Counter
cc = Counter(labels)
durs = np.array([c["cdur"] for c in clips])
print("=== SUMMARY ===")
print(f"clips={len(clips)} clusters={nclust}")
print(f"dur min/med/max = {durs.min():.2f}/{np.median(durs):.2f}/{durs.max():.2f}s")
print("clips per cluster:", dict(sorted(cc.items())))
# tag agreement info: for each cluster, majority subtitle tag
tagmap = {}
for c, lb in zip(clips, labels):
    if c["speaker"]:
        tagmap.setdefault(lb, Counter())[c["speaker"]] += 1
print("cluster -> subtitle tag hints:")
for lb in sorted(tagmap):
    print(f"  cluster_{lb:02d}: {dict(tagmap[lb].most_common(3))}")
print("METADATA:", meta_path)
