#!/bin/bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ttsizer
export PYTHONPATH="$HOME/gsv_pyfix:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PERL5LIB="$HOME/perllib"
OUTROOT="$HOME/subs_out_all/episodes"; mkdir -p "$OUTROOT"

run_one() {
  local n=$1; local gpu=$2
  local ep="ep$(printf "%02d" "$n")"
  local srt="$HOME/frieren_subs/ep$(printf "%02d" "$n").srt"
  local vocals="$HOME/ep_vocals/ep$(printf "%02d" "$n")_vocals.wav"
  local out="$OUTROOT/$ep"
  if [ -f "$out/metadata.csv" ]; then echo "SKIP $ep (done)"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu python "$HOME/extract_episode.py" "$ep" "$srt" "$vocals" "$out" >"$HOME/subs_out_all/${ep}.log" 2>&1
  echo "DONE $ep (gpu$gpu) rc=$? $(date +%H:%M:%S)"
}

# all 28 episodes at once, 7 per GPU (4 GPUs x 7 = 28)
i=0
for n in $(seq 1 28); do
  gpu=$((i % 4))
  run_one "$n" "$gpu" &
  i=$((i+1))
done
wait
echo "PHASE_A_ALL_DONE $(date +%H:%M:%S)"
