import os, sys, glob, csv
os.chdir(os.path.expanduser("~/GPT-SoVITS"))
sys.path.insert(0, os.path.expanduser("~/GPT-SoVITS"))
sys.path.insert(0, os.path.expanduser("~/GPT-SoVITS/GPT_SoVITS"))
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np
import soundfile as sf
from types import SimpleNamespace
import GPT_SoVITS.inference_webui as iw
from pick_clean_ref import pick

OUT = os.path.expanduser("~/qc_test_v2pro_new/batch_selfintro")
JA_L, KO_L = iw.i18n("日文"), iw.i18n("韩文")

# 예시 설정 - 실제 사용 시 대상 캐릭터로 교체
NAME_KO = {
    "ハイター": "하이터", "ファルシュ": "팔슈", "フェルン": "페른", "ユーベル": "위벨",
    "リヒター": "리히터", "レルネン": "레르넨", "グラナト": "그라나토", "シュタルク": "슈타르크",
    "シュトルツ": "슈톨츠",
}
# GPT-SoVITS 자체가 레퍼런스 오디오 3~10초를 강제하는데, pick_clean_ref의 안전마진(0.3s) 후보들이
# 전부 3초 미만인 캐릭터는 자동선택이 후보를 못 찾음 -> 수동으로 3초 이상 클립을 지정
MANUAL_REF = {
    "シュトルツ": "/home/piai_intern/piai_intern2/subs_out_all/final/シュトルツ/ep09_ep09_01029744_01032985.wav",
    "レルネン": "/home/piai_intern/piai_intern2/subs_out_all/final/レルネン/ep18_ep18_00457707_00461624.wav",
}
MAX_TRIES = 6
RATIO_THRESH = 0.3  # 시작0.4s RMS / 전체RMS. 이 미만이면 "생성스킵형"(앞부분 무음/누락)으로 판정

meta = os.path.expanduser("~/subs_out_all/final/metadata_all.csv")
text_of = {r["clip"]: r["jp_text"] for r in csv.DictReader(open(meta, encoding="utf-8"))}

def synth_once(sov_path, gpt_path, ref_wav, ref_text, inp_refs, target):
    for _ in iw.change_sovits_weights(sov_path, prompt_language=JA_L, text_language=KO_L):
        pass
    iw.change_gpt_weights(gpt_path)
    gen = iw.get_tts_wav(
        ref_wav_path=ref_wav, prompt_text=ref_text, prompt_language=JA_L,
        text=target, text_language=KO_L, how_to_cut=iw.i18n("凑四句一切"),
        top_k=15, top_p=1.0, temperature=1.0, ref_free=False,
        speed=1.0, if_freeze=False,
        inp_refs=[SimpleNamespace(name=p) for p in inp_refs] if inp_refs else [],
        sample_steps=8, if_sr=False, pause_second=0.3,
    )
    sr, audio = None, None
    for sr, audio in gen:
        pass
    return sr, audio

def synth_with_retry(label, sov_path, gpt_path, ref_wav, ref_text, inp_refs, target):
    # GPT 자기회귀 생성 자체가 확률적 샘플링이라 같은 입력도 실행마다 결과가 다를 수 있음
    # (레퍼런스가 깨끗해도 가끔 문장 앞부분을 스킵) - 그래서 사후 QA + 재시도가 필요
    best = None
    for attempt in range(1, MAX_TRIES + 1):
        sr, audio = synth_once(sov_path, gpt_path, ref_wav, ref_text, inp_refs, target)
        audio = np.asarray(audio, dtype=np.float64)
        win04 = int(0.4 * sr)
        start_rms = float(np.sqrt(np.mean(audio[:win04] ** 2)))
        overall_rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        ratio = start_rms / overall_rms if overall_rms > 0 else 0.0
        ok = ratio >= RATIO_THRESH
        print(f"  [{label}] try{attempt}: dur={len(audio)/sr:.2f}s ratio={ratio:.3f} {'OK' if ok else 'RETRY'}")
        if best is None or ratio > best[2]:
            best = (sr, audio, ratio)
        if ok:
            return sr, audio, attempt
    print(f"  [{label}] {MAX_TRIES}번 다 기준미달, 그중 제일 나은 것 채택(ratio={best[2]:.3f})")
    return best[0], best[1], MAX_TRIES

for char, name_ko in NAME_KO.items():
    sov_path = sorted(glob.glob(f"SoVITS_weights_v2Pro/{char}_e8_s*.pth"))[-1]
    gpt_path = f"GPT_weights_v2Pro/{char}-e15.ckpt"
    if char in MANUAL_REF:
        ref_wav = MANUAL_REF[char]
        multi = [ref_wav]
    else:
        cands = pick(char, min_dur=2.5, max_dur=8.0, safe_margin=0.3, topn=5)
        if not cands:
            cands = pick(char, min_dur=2.5, max_dur=8.0, safe_margin=0.1, topn=5)
        ref_wav = cands[0][3]
        multi = [c[3] for c in cands[:3]]
    rel = char + "/" + os.path.basename(ref_wav)
    ref_text = text_of.get(rel, "")
    target = f"안녕하세요, 장송의 프리렌의 {name_ko}입니다. 제 한국어가 자연스러운가요?"
    print(f"=== {char} ({name_ko}) ref={os.path.basename(ref_wav)} ===")

    sr, audio, tries = synth_with_retry(f"{char}_single", sov_path, gpt_path, ref_wav, ref_text, [], target)
    sf.write(os.path.join(OUT, f"{char}_single_retry.wav"), audio, sr)

    sr, audio, tries = synth_with_retry(f"{char}_multi", sov_path, gpt_path, ref_wav, ref_text, multi, target)
    sf.write(os.path.join(OUT, f"{char}_multi_retry.wav"), audio, sr)

print("DONE ALL")
