#!/bin/bash
# Stage-3 convergence pilot: Tfull then Tnone, 256x256, 20 x 1000 steps, seed 42, full val every epoch.
# Resumable: re-running continues each model from its last epoch checkpoint (bit-exact, see recovery_test.json).
# Sleep prevention: caffeinate -i (idle) -m (disk) -s (system, on AC power).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
OUT=results/v9_convergence_pilot
L=$OUT/train.log
for c in Tfull Tnone; do
  echo "=== cfg $c start/resume $(date '+%F %T')" >> "$L"
  caffeinate -ims .venv/bin/python experiments/run_ablation.py --config "$c" --seed 42 \
      --epochs 20 --steps_per_epoch 1000 --crop 256 --batch_size 4 --warmup 500 --ema_decay 0.999 \
      --val_n 250 --log_every 200 --vgg_normalize 1 --resume --out_dir "$OUT/seed42" 2>&1 \
      | grep -v -i warn | grep -E "EPOCH|done|resumed|already finished|Error|error|Traceback|non-finite" >> "$L"
  echo "=== cfg $c exit ${PIPESTATUS[0]} $(date '+%F %T')" >> "$L"
done
echo "=== PILOT FINISHED $(date '+%F %T')" >> "$L"
