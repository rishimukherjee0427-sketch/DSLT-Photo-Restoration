# Seed-2026 full-model divergence: diagnosis

Replay of the first 5,000 steps of `Tfull` seeds 2026 (diverged) and 123 (healthy control) with the identical setup
(`experiments/diag_divergence.py`), logging every step. Both replays reproduce the original runs exactly
(per-epoch validation PSNR and training loss match the stored histories for epochs 1-5).

| | seed 2026 (diverged) | seed 123 (control) |
|---|---|---|
| non-finite losses | 0 | 0 |
| max pre-clip gradient norm | 2.4e8 | 6.7 |
| steps with gradient norm > 100 | 3,510 (from step 1,486) | 0 |
| clipping active (norm > 1.0), steps 1,501-5,000 | 100% | 54-58% |
| share of the gradient norm in the bottleneck translator, steps 1,501-5,000 | 1.00 | 0.01-0.02 |

Sequence (seed 2026):
1. Steps 1-1,000 look like the control (median gradient norm 0.7-1.0, translator ratios ~1).
2. From about step 1,100 the bottleneck translator's relative update ||T(z)-z||/||z|| starts to spike
   (max 34 by step 1,200, 2e8 by step 1,400): the bottleneck feature z feeding it becomes tiny relative to the
   translator's output.
3. From step 1,486 the gradient of the bottleneck translator explodes to 1e6-1e8 on almost every step.
4. Clipping is global (one norm over all parameters), so every other module's update is scaled by ~1e-7: the
   encoder, decoder and other translators effectively stop learning. Training loss plateaus and then rises
   (0.29 -> 0.38 over steps 3,500-5,000), matching the validation drop in the original run.

Interpretation (hypothesis, not directly measured): the translator blocks apply InstanceNorm over 16x16 bottleneck
maps; normalising near-constant channels amplifies gradients by up to ~1/sqrt(eps) per norm layer. The healthy
control also shows large bottleneck ratios (median 4-15 after step 1,500) without blow-up, so the bottleneck
translator is near this regime in normal training too.

Implications: the failure is not a NaN/overflow and is not prevented by clipping; global-norm clipping is what
propagates one module's explosion into a network-wide stall. Candidate stabilisers (untested): per-module or
adaptive gradient clipping, removing InstanceNorm / raising its eps in the bottleneck translator, lower learning
rate for the translators.
