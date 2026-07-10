import os
import sys
import re
import difflib
import asyncio
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

VOICEBOX_ROOT = r"c:\Users\piai\Desktop\new_anti\voicebox-main\voicebox-main"
if VOICEBOX_ROOT not in sys.path:
    sys.path.insert(0, VOICEBOX_ROOT)

# Set database path to VoiceBox's data dir
from backend import config
config.set_data_dir(os.path.join(VOICEBOX_ROOT, "data"))

from backend.database import init_db, get_db
from backend.database.models import VoiceProfile as DBVoiceProfile
from backend.services.profiles import create_profile, add_profile_sample
from backend.models import VoiceProfileCreate


init_db()

SUBTITLE_DIR = r"C:\Users\piai\Desktop\dialect_and_survey_backup\frieren_sub_netflix"

# Whisper output dict from previous run to match subtitle database
WHISPER_TRANSCRIPTS = {
    # Eisen
    "アイゼン_EP03_idx103.wav": "テレルところじゃないだろ ソールが覚えられる",
    "アイゼン_EP04_idx143_merged.wav": "人は死んだら 死んだら 無に帰る",
    "アイゼン_EP04_idx201.wav": "本物の主機はフォルボンチのどこかにある",
    "アイゼン_EP04_idx305_merged.wav": "あなたは、私のお客の最北端だからな。",
    "アイゼン_EP05_idx196.wav": "探しているんだろう リーゲル教国属にある村に主とるくという選手がいる",
    "アイゼン_EP05_idx333.wav": "あらあいつには俺の全てを叩き込んだ",
    "アイゼン_EP12_idx14.wav": "大様はどうか10枚しかくれなかったの。",
    "アイゼン_EP23_idx97.wav": "俺たちの目的は最深分にいるまものだ。",
    
    # Stark
    "シュタルク_EP06_idx216.wav": "時は書きだったから全部くえなくて 視聴と開けたんだよなぁ",
    "シュタルク_EP06_idx235_merged.wav": "何で残ってんの? わかったよ 半合ったよ 半分あげるよそうではなくて",
    "シュタルク_EP07_idx28.wav": "そうじゃなくて魔法で場所は向こう側に運んだりできるだろう",
    "シュタルク_EP07_idx55.wav": "こういうのじゃなくて3つけとか グンズケとかいろいろあるでしょ",
    "シュタルク_EP14_idx13.wav": "っていいだろういつも気付か当たり上がってそんなに俺のことが嫌いかよ",
    "シュタルク_EP18_idx19.wav": "どうして北部公園に入るのに、そんなすぎや魔法使い의 동향이 필요んだ?",
    "シュタルク_EP22_idx298.wav": "ちょっと…いやな…もう…なに?",
    "シュタルク_EP27_idx118.wav": "わからおうとするのが大事だと思うんだよフリー連は頑張っていると思うぜ"
}

def load_srt_texts(srt_path):
    if not os.path.exists(srt_path):
        return []
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    if not blocks or (len(blocks) == 1 and not blocks[0].strip()):
        blocks = content.strip().split('\r\n\r\n')
        
    texts = []
    for block in blocks:
        lines = [l.strip('\r') for l in block.strip().split('\n')]
        if not lines:
            continue
        text_lines = []
        for i, line in enumerate(lines):
            if '-->' in line:
                text_lines = lines[i+1:]
                break
        if not text_lines:
            continue
        raw_text = ' '.join(text_lines).strip()
        text = re.sub(r'\{\\an\d+\}', '', raw_text).strip()
        text = re.sub(r'^（[^）]+）\s*', '', text).strip()
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'<[^>]*>', '', text)
        text = text.replace('\r', '').replace('\n', ' ').strip()
        if text:
            texts.append(text)
    return texts

def find_best_subtitle_match(ep_num, whisper_text):
    srt_pattern = f"S01E{ep_num:02d}"
    srt_file = None
    for f in os.listdir(SUBTITLE_DIR):
        if f.endswith(".srt") and srt_pattern in f:
            srt_file = os.path.join(SUBTITLE_DIR, f)
            break
    if not srt_file:
        return whisper_text
        
    sub_texts = load_srt_texts(srt_file)
    
    # Use sequence matcher to find the best match or combined consecutive matches
    best_ratio = 0.0
    best_match = whisper_text
    
    # Check single subtitles
    for txt in sub_texts:
        ratio = difflib.SequenceMatcher(None, whisper_text, txt).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = txt
            
    # Check pairs of consecutive subtitles
    for i in range(len(sub_texts) - 1):
        combined = sub_texts[i] + " " + sub_texts[i+1]
        ratio = difflib.SequenceMatcher(None, whisper_text, combined).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = combined
            
    return best_match

async def clean_and_register_profile_db(char_name, folder_path, excluded_files, target_profile_name, db):
    print(f"\nProcessing {char_name}...")
    wav_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])
    
    registered_samples = []
    for f in wav_files:
        if f in excluded_files:
            print(f"  Excluding file: {f}")
            continue
            
        match = re.search(r'EP(\d+)', f)
        if not match:
            continue
        ep_num = int(match.group(1))
        
        whisper_text = WHISPER_TRANSCRIPTS.get(f, "")
        if not whisper_text:
            continue
            
        # Match with official Netflix subtitle
        official_text = find_best_subtitle_match(ep_num, whisper_text)
        official_text = re.sub(r'^[（(][^）)]+[）)]\s*', '', official_text).strip()
        
        print(f"  File: {f}")
        print(f"    Whisper: '{whisper_text}'")
        print(f"    Netflix: '{official_text}'")
        
        registered_samples.append((f, official_text))
        
    # Delete existing profile if present
    print(f"\nRegistering Voice Profile: '{target_profile_name}' directly to SQLite DB...")
    try:
        existing = db.query(DBVoiceProfile).filter(DBVoiceProfile.name == target_profile_name).first()
        if existing:
            print(f"  Deleting existing profile: {target_profile_name} (ID: {existing.id})")
            db.delete(existing)
            db.commit()
    except Exception as e:
        print(f"  Warning cleaning DB profile: {e}")
        db.rollback()
        
    # Create new profile
    profile_create = VoiceProfileCreate(
        name=target_profile_name,
        description=f"Refined voice profile for {char_name}",
        language="ja",
        voice_type="cloned",
        default_engine="qwen"
    )
    
    try:
        profile = await create_profile(profile_create, db)
        profile_id = profile.id
        print(f"  Created Profile ID: {profile_id}")

        
        # Upload samples using add_profile_sample
        uploaded_count = 0
        for f, text in registered_samples:
            wav_path = os.path.join(folder_path, f)
            await add_profile_sample(profile_id, wav_path, text, db)
            print(f"    Added sample {f} -> '{text}'")
            uploaded_count += 1
        print(f"  Successfully registered {uploaded_count} samples for {target_profile_name} in DB!")
        return profile_id
    except Exception as e:
        print(f"  Error registering profile in DB: {e}")
        db.rollback()
        return None

async def async_main():
    db = next(get_db())
    
    # 1. Eisen (Exclude EP04_idx305_merged.wav and EP03_idx103.wav)
    eisen_folder = r"D:\anime\zeroshot_dataset\3_cleaned_dataset\5_アイゼン"
    eisen_exclude = ["아이젠_EP04_idx305_merged.wav", "アイゼン_EP04_idx305_merged.wav", "아이젠_EP03_idx103.wav", "アイゼン_EP03_idx103.wav"]
    eisen_id = await clean_and_register_profile_db("Eisen", eisen_folder, eisen_exclude, "아이젠_정제최종", db)
    
    # 2. Stark (Exclude EP22_idx298.wav because it is too short and mumbled 'ちょっと…いやな…もう…なに?')
    stark_folder = r"D:\anime\zeroshot_dataset\3_cleaned_dataset\6_シュタルク"
    stark_exclude = ["슈탈크_EP22_idx298.wav", "シュタルク_EP22_idx298.wav"]
    stark_id = await clean_and_register_profile_db("Stark", stark_folder, stark_exclude, "슈타르크_정제최종", db)

if __name__ == "__main__":
    asyncio.run(async_main())
