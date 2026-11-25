#!/usr/bin/env bash
# Task 3 baselines (U-Net, then NAFNet), Stage-3 protocol, seed 42. Waits for the divergence diagnostic to finish
# (one MPS job at a time), times each model briefly (50 steps, small val), then trains 20 x 1000 steps.
# Resumable: re-running continues from {cfg}_last.pth.
cd "$(dirname "$0")/.."
OUT=results/v10_baselines
L=$OUT/train.log
mkdir -p "$OUT/timing"
D=results/v9_convergence_pilot/diag/diag.log
while [ -f "$D" ] && ! grep -qE "DIAG FINISHED|Traceback|exit [1-9]" "$D"; do sleep 120; done
echo "=== launch $(date '+%F %T') git $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')" >> "$L"
for c in unet nafnet; do
    if [ ! -f "$OUT/timing/${c}_history.json" ]; then
        echo "=== timing $c $(date '+%F %T')" >> "$L"
        caffeinate -ims .venv/bin/python experiments/run_baseline_s3.py --config "$c" --seed 42 --epochs 1 \
            --steps_per_epoch 50 --crop 256 --batch_size 4 --warmup 500 --ema_decay 0.999 --val_n 8 --log_every 25 \
            --vgg_normalize 1 --out_dir "$OUT/timing" 2>&1 | grep -v -i warn | grep -E "step 50/|EPOCH|params|Error|Traceback" >> "$L"
    fi
done
for c in unet nafnet; do
    echo "=== train $c seed 42 start/resume $(date '+%F %T')" >> "$L"
    caffeinate -ims .venv/bin/python experiments/run_baseline_s3.py --config "$c" --seed 42 --epochs 20 \
        --steps_per_epoch 1000 --crop 256 --batch_size 4 --warmup 500 --ema_decay 0.999 --val_n 250 --log_every 200 \
        --vgg_normalize 1 --resume --out_dir "$OUT/seed42" 2>&1 \
        | grep -v -i warn | grep -E "EPOCH|done|resumed|params|Error|error|Traceback|non-finite" >> "$L"
    echo "=== train $c exit ${PIPESTATUS[0]} $(date '+%F %T')" >> "$L"
done
echo "=== BASELINES FINISHED $(date '+%F %T')" >> "$L"
