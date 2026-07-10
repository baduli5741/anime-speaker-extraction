import csv, os, random, numpy as np, soundfile as sf, torch, torchaudio
import wespeaker

OUTDIR = os.path.expanduser("~/subs_out/ep01")  # Path B v1 output (has tagged clips)
WESPK_DIR = os.path.expanduser("~/TTSizer/weights/wespeaker-voxceleb-resnet293-LM")
SR = 16000

wmodel = wespeaker.load_model_local(WESPK_DIR)
try: wmodel.set_device("cuda" if torch.cuda.is_available() else "cpu")
except Exception: pass

def embed(a):
    t = torch.from_numpy(a.astype("float32")).unsqueeze(0)
    e = np.asarray(wmodel.extract_embedding_from_pcm(t, SR), dtype="float32")
    return e / (np.linalg.norm(e) + 1e-8)

def early_late_cos(seg):
    n = len(seg)
    a = seg[:int(0.42*n)]; b = seg[int(0.58*n):]
    if len(a) < int(0.35*SR) or len(b) < int(0.35*SR):
        return None
    return float(np.dot(embed(a), embed(b)))

def load_clip(rel):
    p = os.path.join(OUTDIR, rel)
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
    return a

rows = list(csv.DictReader(open(os.path.join(OUTDIR, "metadata.csv"), encoding="utf-8")))
tagged = [r for r in rows if r.get("sub_speaker_tag", "").strip()]
by_tag = {}
for r in tagged:
    by_tag.setdefault(r["sub_speaker_tag"], []).append(r)
print("tagged clips:", len(tagged), "| tags:", {k: len(v) for k, v in by_tag.items()})

random.seed(0)

# 1. SAME-speaker: intra-clip early/late cosine on genuinely single-tagged clips
same_cos = []
for tag, items in by_tag.items():
    for r in items[:20]:
        seg = load_clip(r["clip"])
        c = early_late_cos(seg)
        if c is not None:
            same_cos.append(c)

# 2. DIFFERENT-speaker: concat two clips from two different tags, same test
diff_cos = []
tags = [t for t in by_tag if len(by_tag[t]) >= 1]
pairs = 0
attempts = 0
while pairs < 40 and attempts < 400:
    attempts += 1
    ta, tb = random.sample(tags, 2) if len(tags) >= 2 else (None, None)
    if ta is None:
        break
    ra, rb = random.choice(by_tag[ta]), random.choice(by_tag[tb])
    seg = np.concatenate([load_clip(ra["clip"]), load_clip(rb["clip"])])
    c = early_late_cos(seg)
    if c is not None:
        diff_cos.append(c); pairs += 1

same_cos = np.array(same_cos); diff_cos = np.array(diff_cos)
print(f"\nSAME-speaker  n={len(same_cos)}  mean={same_cos.mean():.3f} std={same_cos.std():.3f} "
      f"p5={np.percentile(same_cos,5):.3f} p50={np.percentile(same_cos,50):.3f} min={same_cos.min():.3f}")
print(f"DIFF-speaker  n={len(diff_cos)}  mean={diff_cos.mean():.3f} std={diff_cos.std():.3f} "
      f"p50={np.percentile(diff_cos,50):.3f} p95={np.percentile(diff_cos,95):.3f} max={diff_cos.max():.3f}")

overlap = (np.percentile(same_cos, 5) < np.percentile(diff_cos, 95))
print(f"\nDistributions overlap: {overlap}")
thresh = (np.percentile(same_cos, 5) + np.percentile(diff_cos, 95)) / 2
print(f"SUGGESTED THRESHOLD (midpoint same-p5 / diff-p95): {thresh:.3f}")
sep = np.percentile(same_cos, 5) - np.percentile(diff_cos, 95)
print(f"Separation margin: {sep:.3f}  ({'usable' if sep > 0.03 else 'POOR separation -- filter may not be reliable'})")
