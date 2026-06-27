#!/usr/bin/env bash
# Standard fair-eval protocol watcher (wiki pi05-openpi-deployment "Standard fair-eval protocol").
# Waits for each 10K checkpoint of a run, then scores it TWICE with eval_pika_umi_val_tcp_lerobot.py:
#   (1) held-out 86 val   (2) train subset (--limit 86)  -> TRAIN-vs-VAL per-axis diagnostic.
# Runs pinned to one GPU with on-demand memory so the training rig (other GPUs) is untouched.
set -u
cd /home/plaif/workspace/pi
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME=/home/plaif/workspace/lerobot_home
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-7}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10

CONFIG=pi05_pika_umi_video_tcp_gripabs_velproprio_depth_z50_h24
CKPT_BASE=/home/plaif/workspace/openpi_runs/checkpoints/${CONFIG}/depth_z50_real_h24
VAL=plaif/pika_umi_video_val_tcp_gripabs_velproprio_depth_z50
TRAIN=plaif/pika_umi_video_train_tcp_gripabs_velproprio_depth_z50

run_eval () {  # $1=step $2=repo $3=tag $4=extra
  # Retry on GPU OOM: GPU 7 is shared with training (~9.7GB free), so a depth eval can transiently
  # collide with a training memory spike (BLAS-init / RESOURCE_EXHAUSTED). Retry a few times.
  local log="/tmp/fair_eval_step$1_$3.log"
  for attempt in 1 2 3 4 5; do
    echo ">>> eval step $1 ($3) attempt $attempt $(date)"
    uv run python examples/pika_umi/eval_pika_umi_val_tcp_lerobot.py \
      --checkpoint-step "$1" --config "$CONFIG" --ckpt-base "$CKPT_BASE" \
      --val-repo-id "$2" --lerobot-home /home/plaif/workspace/lerobot_home \
      --gripper-action absolute --tag "$3" $4 > "$log" 2>&1
    if grep -q "wrote " "$log"; then echo "    OK -> $log"; return 0; fi
    echo "    failed ($(grep -aoE 'BLAS|RESOURCE_EXHAUSTED' "$log" | head -1)); retry in 90s"; sleep 90
  done
  echo "    GAVE UP after 5 attempts -> $log"
}

for STEP in 10000 20000 30000 40000; do
  echo "=== waiting for checkpoint $STEP ==="
  until [ -d "$CKPT_BASE/$STEP/params" ]; do sleep 60; done
  sleep 20  # let the async checkpoint finalize
  OUT=/home/plaif/workspace/openpi_runs/eval_val_tcp_lerobot
  [ -f "$OUT/step${STEP}_z50real_val${STEP}.json" ]   || run_eval "$STEP" "$VAL"   "z50real_val${STEP}"   ""
  [ -f "$OUT/step${STEP}_z50real_train${STEP}.json" ] || run_eval "$STEP" "$TRAIN" "z50real_train${STEP}" "--limit 86"
done
echo "=== FAIR-EVAL WATCHER DONE $(date) ==="
