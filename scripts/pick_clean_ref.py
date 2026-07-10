import os, re, sys, glob, bisect
import soundfile as sf

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

def pick(char, min_dur=3.0, max_dur=8.0, safe_margin=0.3, topn=5):
    files = glob.glob(os.path.expanduser(f"~/subs_out_all/final/{char}/*.wav"))
    cache = {}
    cands = []
    for f in files:
        fn = os.path.basename(f)
        m = re.match(r"(ep\d+)_ep\d+_(\d+)_(\d+)\.wav", fn)
        if not m:
            continue
        ep, s_ms, e_ms = m.group(1), int(m.group(2)), int(m.group(3))
        s, e = s_ms/1000, e_ms/1000
        dur = e - s
        if not (min_dur <= dur <= max_dur):
            continue
        if ep not in cache:
            cache[ep] = all_bounds(ep)
        starts, ends = cache[ep]
        i = bisect.bisect_left(starts, e)  # exact-equal boundary must count as gap=0
        next_start = starts[i] if i < len(starts) else 999999
        j = bisect.bisect_right(ends, s)   # exact-equal boundary must count as gap=0
        prev_end = ends[j-1] if j > 0 else -1
        gap_next = next_start - e
        gap_prev = s - prev_end
        # 경계 깨끗 + 길이 조건 만족하는 것만
        if gap_next >= safe_margin and gap_prev >= safe_margin:
            cands.append((dur, gap_next, gap_prev, f))
    cands.sort(key=lambda x: -x[0])  # 길이 내림차순
    return cands[:topn]

if __name__ == "__main__":
    char = sys.argv[1]
    results = pick(char)
    if not results:
        print(f"[{char}] 안전마진(>=0.3s) 만족하는 후보 없음 - margin 낮춰야 함")
    for dur, gn, gp, f in results:
        print(f"dur={dur:.2f}s gap_next={gn:.2f}s gap_prev={gp:.2f}s  {f}")
