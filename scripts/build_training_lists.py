import csv, os, re, shutil
from collections import Counter

SRC_META = os.path.expanduser("~/subs_out_all/final/metadata_all.csv")
SRC_ROOT = os.path.expanduser("~/subs_out_all/final")
OUT_DIR = os.path.expanduser("~/gptsovits_data_v2/lists")
os.makedirs(OUT_DIR, exist_ok=True)

# 통칭/제네릭 태그 제외 (4개 화 이상 분산 = 매번 다른 사람으로 판정)
GENERIC_EXCLUDE = {"子供", "衛兵", "商人", "盗賊", "２人", "少女", "uncertain",
                   "村人", "老人", "衛兵隊長"}  # 村人/老人/衛兵隊長은 클립 0~안전마진용 방어적 포함

MIN_CLIPS = 20  # 학습 최소 클립수 (위벨 22개로도 v2Pro 학습 성공한 전례 참고)

FURIGANA_RE = re.compile(r"([一-龯々]+)\(([ぁ-んー]+)\)")  # 한자(히라가나읽기)
ELLIPSIS_RE = re.compile(r"…")
MUSIC_RE = re.compile(r"♪")

def clean_text(t):
    # 후리가나: 한자 버리고 히라가나 읽기만 남김
    t = FURIGANA_RE.sub(lambda m: m.group(2), t)
    # 말줄임표 -> 쉼표
    t = ELLIPSIS_RE.sub("、", t)
    return t.strip()

rows = list(csv.DictReader(open(SRC_META, encoding="utf-8")))
print(f"원본 총 {len(rows)}행")

by_char = {}
excluded_music = 0
excluded_generic = 0
for r in rows:
    ch = r["clip"].split("/")[0]
    text = r["jp_text"]
    if MUSIC_RE.search(text):
        excluded_music += 1
        continue
    if ch in GENERIC_EXCLUDE:
        excluded_generic += 1
        continue
    cleaned = clean_text(text)
    if not cleaned:
        continue
    by_char.setdefault(ch, []).append((r["clip"], cleaned))

print(f"제외: 음악{excluded_music}건, 통칭태그{excluded_generic}건")
print(f"정제후 캐릭터 후보: {len(by_char)}명")

qualified = {ch: items for ch, items in by_char.items() if len(items) >= MIN_CLIPS}
print(f"\nMIN_CLIPS={MIN_CLIPS} 이상 캐릭터: {len(qualified)}명")

for ch in sorted(qualified, key=lambda x: -len(qualified[x])):
    items = qualified[ch]
    list_path = os.path.join(OUT_DIR, f"{ch}.list")
    with open(list_path, "w", encoding="utf-8") as f:
        for clip_rel, text in items:
            abs_wav = os.path.join(SRC_ROOT, clip_rel)
            f.write(f"{abs_wav}|{ch}|ja|{text}\n")
    print(f"  {ch}: {len(items)}줄 -> {list_path}")

dropped = {ch: len(items) for ch, items in by_char.items() if len(items) < MIN_CLIPS}
if dropped:
    print(f"\nMIN_CLIPS 미달로 제외됨({len(dropped)}명): {dropped}")
