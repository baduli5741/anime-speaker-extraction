import os, re, glob, bisect

def ts(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def all_bounds(ep):
    path = os.path.expanduser(f"~/frieren_subs/{ep}.srt")
    raw = open(path, encoding="utf-8-sig").read()
    blocks = re.split(r"\r?\n\r?\n", raw.strip())
    starts, ends = [], []
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        tl = next((l for l in lines if "-->" in l), None)
        if not tl:
            continue
        s, e = [ts(x.strip()) for x in tl.split("-->")]
        starts.append(s); ends.append(e)
    return sorted(starts), sorted(ends)

BOUND_CACHE = {}
def gaps(ep, s, e):
    if ep not in BOUND_CACHE:
        BOUND_CACHE[ep] = all_bounds(ep)
    starts, ends = BOUND_CACHE[ep]
    i = bisect.bisect_left(starts, e)
    next_start = starts[i] if i < len(starts) else 999999
    j = bisect.bisect_right(ends, s)
    prev_end = ends[j-1] if j > 0 else -1
    return next_start - e, s - prev_end

SAFE = 0.3
base = os.path.expanduser("~/subs_out_all/final")
chars = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))

risky = []
for char in chars:
    files = glob.glob(os.path.join(base, char, "*.wav"))
    cands = []
    for f in files:
        fn = os.path.basename(f)
        m = re.match(r"(ep\d+)_ep\d+_(\d+)_(\d+)\.wav", fn)
        if not m:
            continue
        ep, s_ms, e_ms = m.group(1), int(m.group(2)), int(m.group(3))
        s, e = s_ms/1000, e_ms/1000
        dur = e - s
        if 4.0 <= dur <= 8.0:
            cands.append((dur, ep, s, e, f))
    if not cands:
        continue
    cands.sort(key=lambda x: -x[0])
    dur, ep, s, e, f = cands[0]  # 기존 방식이 골랐을 1등
    gn, gp = gaps(ep, s, e)
    flag = "OK" if (gn >= SAFE and gp >= SAFE) else "위험"
    if flag == "위험":
        risky.append((char, dur, gn, gp, f))

print(f"총 {len(chars)}명 캐릭터 스캔, 위험(경계오염) {len(risky)}명:")
for char, dur, gn, gp, f in risky:
    print(f"  {char}: dur={dur:.2f}s gap_next={gn:.2f}s gap_prev={gp:.2f}s  {os.path.basename(f)}")
