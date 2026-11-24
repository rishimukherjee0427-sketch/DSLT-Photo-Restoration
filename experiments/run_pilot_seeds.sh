#!/bin/bash
# Stage-3 convergence pilot, additional seeds (123, 2026). Identical to run_pilot.sh (seed 42) except --seed / out dir / log.
# Tfull then Tnone per seed, 256x256, 20 x 1000 steps, full val every epoch, FP32.
# Resumable: re-running continues each model from its last epoch checkpoint (bit-exact, see recovery_test.json);
# finished models print "already finished" and are skipped. Never touches results/v9_convergence_pilot/seed42.
# Sleep prevention: caffeinate -i (idle) -m (disk) -s (system, on AC power).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
OUT=results/v9_convergence_pilot
L=$OUT/train_seeds123_2026.log
echo "=== launch $(date '+%F %T') git $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')" >> "$L"
for s in 123 2026; do
  for c in Tfull Tnone; do
    echo "=== seed $s cfg $c start/resume $(date '+%F %T')" >> "$L"
    caffeinate -ims .venv/bin/python experiments/run_ablation.py --config "$c" --seed "$s" \
        --epochs 20 --steps_per_epoch 1000 --crop 256 --batch_size 4 --warmup 500 --ema_decay 0.999 \
        --val_n 250 --log_every 200 --vgg_normalize 1 --resume --out_dir "$OUT/seed$s" 2>&1 \
        | grep -v -i warn | grep -E "EPOCH|done|resumed|already finished|Error|error|Traceback|non-finite" >> "$L"
    echo "=== seed $s cfg $c exit ${PIPESTATUS[0]} $(date '+%F %T')" >> "$L"
  done
done
echo "=== SEEDS FINISHED $(date '+%F %T')" >> "$L"
