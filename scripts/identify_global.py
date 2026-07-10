#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase B: global speaker identity, run ONCE over ALL episodes' pooled clips.
This is what keeps character identity consistent across the 28 episodes
(per-episode clustering would give unrelated numbering per episode).

Strategy:
1. Pool every episode's embeddings.npz + metadata.csv.
2. Build one centroid per subtitle-tagged character name, pooled across all
   episodes (more tagged examples -> more stable centroid than a single ep).
3. Assign EVERY clip (tagged or not) to its nearest centroid if similarity
   clears a threshold; else "uncertain" bucket for manual naming.
4. Flag tagged clips whose own embedding disagrees with their tag's centroid
   (embedding closer to a DIFFERENT character's centroid) as likely
   mislabeled subtitle tags -- useful QA signal.

No GPU needed (works on cached embeddings only).
"""
import os, csv, glob, shutil, sys
import numpy as np
from collections import defaultdict, Counter

EXTRACT_ROOT = os.path.expanduser("~/subs_out_all/episodes")
OUT_ROOT = os.path.expanduser("~/subs_out_all/final")
ASSIGN_THRESH = float(os.environ.get("ASSIGN_THRESH", "0.35"))  # cosine dist to centroid

os.makedirs(OUT_ROOT, exist_ok=True)

all_rows = []   # each: dict with clip path (abs), ep, tag, text, emb (np array)
tag_embs = defaultdict(list)

ep_dirs = sorted(glob.glob(os.path.join(EXTRACT_ROOT, "ep*")))
print(f"Found {len(ep_dirs)} episode dirs")

for ep_dir in ep_dirs:
    meta_path = os.path.join(ep_dir, "metadata.csv")
    emb_path = os.path.join(ep_dir, "embeddings.npz")
    if not (os.path.isfile(meta_path) and os.path.isfile(emb_path)):
        continue
    rows = list(csv.DictReader(open(meta_path, encoding="utf-8")))
    embs = np.load(emb_path)["embs"]
    if len(rows) != len(embs):
        print(f"  WARN {ep_dir}: metadata/embeddings length mismatch ({len(rows)} vs {len(embs)}), skipping")
        continue
    for r, e in zip(rows, embs):
        r["_emb"] = e
        r["_abs_wav"] = os.path.join(ep_dir, r["clip"])
        all_rows.append(r)
        tag = r.get("sub_speaker_tag", "").strip()
        if tag:
            tag_embs[tag].append(e)

print(f"Total clips pooled: {len(all_rows)}")
print(f"Tagged clips: {sum(len(v) for v in tag_embs.values())}  across {len(tag_embs)} tags")

# Build centroids (L2-normalized mean of tagged embeddings)
centroids = {}
for tag, es in tag_embs.items():
    if len(es) < 3:
        continue  # too few examples to trust a centroid
    c = np.mean(np.vstack(es), axis=0)
    centroids[tag] = c / (np.linalg.norm(c) + 1e-8)
print(f"Centroids built for {len(centroids)} characters (>=3 tagged clips): {list(centroids.keys())}")

names = list(centroids.keys())
C = np.vstack([centroids[n] for n in names]) if names else np.zeros((0, 256))

def best_match(emb):
    if len(names) == 0:
        return None, 0.0
    sims = C @ emb
    i = int(np.argmax(sims))
    return names[i], float(sims[i])

# Assign every clip + QA flag for tag disagreement
assign_counts = Counter()
mismatch_flags = []
final_rows = []
for r in all_rows:
    emb = r["_emb"]
    match, sim = best_match(emb)
    tag = r.get("sub_speaker_tag", "").strip()
    if match is not None and sim >= (1 - ASSIGN_THRESH):
        assigned = match
    else:
        assigned = "uncertain"
    assign_counts[assigned] += 1
    if tag and match is not None and tag != match and sim >= (1 - ASSIGN_THRESH):
        mismatch_flags.append((r["_abs_wav"], tag, match, sim))
    final_rows.append((r, assigned, sim))

print("\nAssignment counts:")
for name, n in assign_counts.most_common():
    print(f"  {name}: {n}")

print(f"\nTag/embedding mismatches (possible subtitle mislabel): {len(mismatch_flags)}")
for path, tag, match, sim in mismatch_flags[:20]:
    print(f"  {os.path.basename(path)}: tag={tag} but closest={match} (sim={sim:.3f})")

# Write final folders + combined metadata
combined_meta = os.path.join(OUT_ROOT, "metadata_all.csv")
with open(combined_meta, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["clip", "assigned", "sim", "ep", "start", "end", "dur", "jp_text", "sub_speaker_tag"])
    for r, assigned, sim in final_rows:
        dest_dir = os.path.join(OUT_ROOT, assigned)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{r['ep']}_{os.path.basename(r['_abs_wav'])}")
        shutil.copy(r["_abs_wav"], dest)
        w.writerow([os.path.relpath(dest, OUT_ROOT), assigned, f"{sim:.3f}", r["ep"],
                    r["start"], r["end"], r["dur"], r["jp_text"], r.get("sub_speaker_tag", "")])

print(f"\nFinal output: {OUT_ROOT}")
print(f"Combined metadata: {combined_meta}")
