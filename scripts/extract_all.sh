#!/bin/bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ttsizer
export PYTHONPATH="$HOME/gsv_pyfix:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PERL5LIB="$HOME/perllib"
OUTROOT="$HOME/subs_out_all/episodes"; mkdir -p "$OUTROOT"

worker() {
  local gpu=$1; shift
  export CUDA_VISIBLE_DEVICES=$gpu
  for n in "$@"; do
    ep="ep$(printf "%02d" "$n")"
    srt="$HOME/frieren_subs/ep$(printf "%02d" "$n").srt"
    vocals="$HOME/ep_vocals/ep$(printf "%02d" "$n")_vocals.wav"
    out="$OUTROOT/$ep"
    if [ -f "$out/metadata.csv" ]; then echo "[GPU$gpu] SKIP $ep (already done)"; continue; fi
    echo "[GPU$gpu] START $ep $(date +%H:%M:%S)"
    python "$HOME/extract_episode.py" "$ep" "$srt" "$vocals" "$out" >"$HOME/subs_out_all/${ep}.log" 2>&1
    echo "[GPU$gpu] DONE  $ep rc=$? $(date +%H:%M:%S)"
  done
}

g0=(); g1=(); g2=(); i=0
for n in $(seq 1 28); do case $((i%3)) in 0)g0+=("$n");; 1)g1+=("$n");; 2)g2+=("$n");; esac; i=$((i+1)); done
worker 0 "${g0[@]}" &
worker 1 "${g1[@]}" &
worker 2 "${g2[@]}" &
wait
echo "PHASE_A_ALL_DONE $(date +%H:%M:%S)"
