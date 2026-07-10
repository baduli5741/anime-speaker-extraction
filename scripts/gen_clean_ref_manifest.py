import os, csv, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick_clean_ref import pick

meta = os.path.expanduser("~/subs_out_all/final/metadata_all.csv")
text_of = {r["clip"]: r["jp_text"] for r in csv.DictReader(open(meta, encoding="utf-8"))}

base = os.path.expanduser("~/subs_out_all/final")
chars = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))

manifest = {}
no_safe_candidate = []
for char in chars:
    cands = pick(char, min_dur=3.0, max_dur=8.0, safe_margin=0.3, topn=1)
    if not cands:
        no_safe_candidate.append(char)
        continue
    dur, gn, gp, f = cands[0]
    rel = char + "/" + os.path.basename(f)
    manifest[char] = {"wav": f, "dur": round(dur, 2), "text": text_of.get(rel, "")}

out = os.path.expanduser("~/clean_ref_manifest.json")
json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"생성완료: {len(manifest)}명, 안전후보 없음: {len(no_safe_candidate)}명")
print("안전후보 없는 캐릭터(마진 낮춰서 재검토 필요):", no_safe_candidate)
