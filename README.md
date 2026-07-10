# anime-speaker-extraction

자막 있는 애니메이션에서 **캐릭터별로 깨끗한 (오디오, 텍스트) 클립**을 뽑아내는 로컬 파이프라인. TTS 파인튜닝(GPT-SoVITS 등) 학습데이터 제작 목적으로 만들었고, 클라우드 API 없이 완전 로컬로 돈다.

## 왜 만들었나

애니 음성으로 TTS를 학습시키려면 캐릭터별로 "이 오디오 = 이 텍스트, 다른 사람 목소리 안 섞임"인 클립이 필요하다. 자막(SRT)에 텍스트와 화자 이름이 일부 있지만:
- 자막 타임스탬프는 가독성을 위해 늘려 잡혀있어(업계 표준 0.5~2초 연장) 그대로 자르면 다음 화자를 물어버림
- 화자 태그가 없는 줄이 많고, 이름-임베딩 매칭(임계값 방식)은 애매하면 대량 유실되거나 비슷한 목소리끼리 오매칭됨

## 파이프라인

**Phase A — 화별 추출** (`scripts/extract_episode.py`, GPU 필요)
1. SRT 파싱, 화자태그 2개 이상 박힌 줄(다중화자) 드롭
2. CTC 강제정렬(ctc-forced-aligner, 일본어)로 자막 텍스트를 오디오에 재정렬
3. Silero VAD로 무음경계까지 확장(말 잘림 방지)
4. **원본 SRT 전체(필터링 전) 기준 이웃경계 하드클램프** — 다음/이전 화자 자막을 절대 침범 못 하게 강제
5. Wespeaker(ResNet293) 임베딩 추출
6. 클립 + 임베딩 + 메타데이터 저장 (클러스터링은 안 함)

**Phase B — 전체 화자 식별** (`scripts/identify_global.py`, GPU 불필요, 캐시된 임베딩만 사용)
1. 모든 화의 클립을 하나로 풀링
2. 자막 태그 있는 클립들로 캐릭터별 대표벡터(centroid) 생성 (전체 화 통합 → 안정적)
3. 모든 클립을 최근접 centroid에 배정, 애매하면 `uncertain/`로 보류
4. 자막태그와 임베딩판정이 어긋나면 QA 플래그(자막 오라벨 의심)

화별로 따로 클러스터링하면 화마다 번호가 뒤죽박죽이라 **정체성 부여는 전체 통합 후 한 번만** 한다.

## 셋업 (오래된 GPU에서 중요)

- **cuDNN 비활성 셔우 필수** — V100(SM7.0)·P40(SM6.1) 둘 다 최신 cuDNN(9.x, SM≥7.5 요구)과 안 맞아 conv 커널이 죽는다. `gsv_pyfix/sitecustomize.py`를 `PYTHONPATH`에 추가하면 자동으로 `torch.backends.cudnn.enabled=False` 적용.
- **perllib 필요** — `ctc-forced-aligner`의 uroman(일본어 로마자화)이 일부 노드의 손상된 perl(`FindBin.pm` 없음)에서 죽는다. `perllib/FindBin.pm` 스텁을 `PERL5LIB`에 추가.
- 언어코드는 `"jpn"`(ISO-639-3), `"ja"` 아님 — 아니면 uroman이 한 줄 전체를 뭉텅이로 처리하다 죽음.

```bash
export PYTHONPATH="$PWD/gsv_pyfix:$PYTHONPATH"
export PERL5LIB="$PWD/perllib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 실행

```bash
# Phase A: 화 하나
python scripts/extract_episode.py ep01 subs/ep01.srt vocals/ep01_vocals.wav out/episodes/ep01

# Phase A: 여러 화 병렬 (예시 스크립트, GPU 수/화 수에 맞게 조정)
bash scripts/extract_all.sh          # GPU당 1워커, 순차 루프
bash scripts/extract_all_p40.sh      # GPU당 여러워커, 전체 동시실행 (VRAM 여유있을 때)

# Phase B: 전체 화자 식별 (한 번만, 모든 화 끝난 뒤)
python scripts/identify_global.py
```

## 알려진 한계 (미해결)

경계클램프(v6)로 대부분의 잘림/오염을 잡았지만, **대사가 빠르게 오가는 구간에서 소량 잔여 누출**이 있다:
- 고정 안전마진(0.05s)이 너무 좁아 다음화자 실제발화가 자막시각보다 먼저 시작하면 못 막음
- VAD 경계확장이 실제 무음갭이 거의 없는 구간에서 자막경계 도달 전까지 계속 늘어남

자막·VAD 둘 다 신뢰 부족한 케이스라, **다음 시도는 순수 음향 최소점(에너지/VAD확률 국소최소)에 스냅**하는 방식. 자세한 배경은 `EXPERIMENT_LOG.md` 참고.

## 참고했지만 채택 안 한 도구

- [TTSizer](https://github.com/taresh18/TTSizer) — 애니 TTS 데이터셋 자동화(MelBandRoformer+CTC정렬+Wespeaker), 다이어라이제이션이 Gemini API(클라우드) 의존이라 완전로컬 요구사항에 안 맞음. 실험 상세는 로그 참고.
- [AnimeSpeech](https://github.com/deeplearningcafe/animespeechdataset) — 자막기반·로컬·일본어, 화자ID가 임베딩+임계값+KNN이라 우리가 겪은 flaky 매칭 문제와 유사한 접근.

## legacy/

CTC 강제정렬 파이프라인 이전에 시도했던 것들 (참고용, 현재 미사용):
- `build_global_centroids.py` — 임계값 기반 이름-임베딩 매칭. 임계값 딜레마(높이면 대량 `_unknown` 유실, 낮추면 오매칭)로 폐기.
- `frieren_hybrid_extractor.py`, `align_metadata_with_netflix.py`, `vad_slice_poc.py` — Whisper 전사+자막 퍼지매칭+VAD청크 기반 초기 시도.
- `subs_extract_v3.py` — sub-window(클립 반쪼개기) 화자전환 검출 시도, 실측 결과 완전 실패(같은화자/다른화자 코사인분포 분리마진 -0.423). `calibrate_thresh.py`가 그 증거.
- `subs_extract_v6.py` — Phase 분리 전 단일화 버전(화별로 CTC정렬+클램프+클러스터링까지 한 스크립트에서 함). 로직은 `extract_episode.py`에 계승됨.
