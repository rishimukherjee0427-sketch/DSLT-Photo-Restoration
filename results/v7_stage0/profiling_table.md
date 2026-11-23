## Computational profile (Apple M4 GPU via MPS, FP32, inference, torch.no_grad; median of 30 runs after 10 warm-up)

| Model | Params (M) | GMACs @128² | GMACs @256² | Latency ms @128² b1 | Latency ms @256² b1 | Throughput img/s @256² b4 | Peak MPS driver mem GB @256² b4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSLT-A | 44.41 | 19.1 | 76.3 | 21.8 | 57.5 | 24.9 | 2.29 |
| DSLT-B | 46.06 | 20.6 | 82.5 | 24.3 | 68.1 | 20.9 | 2.29 |
| DSLT-C | 46.06 | 20.6 | 82.5 | 24.4 | 68.1 | 20.9 | 2.29 |
| DSLT-O | 46.06 | 20.9 | 83.5 | 25.3 | 73.1 | 19.9 | 2.29 |
| Restormer-tiny | 3.75 | 5.1 | 20.6 | 18.4 | 51.1 | 13.5 | 3.31 |
| Promptir-tiny | 7.22 | 6.4 | 25.7 | 37.3 | 71.1 | 12.1 | 4.38 |

Notes: MACs from torch.utils.flop_counter (convolutions/matmuls; element-wise ops, pooling and normalisation not counted). DSLT-A excludes the damage estimator and unused gate modulators (not executed at inference). DSLT-O = DSLT-C compute plus a second pass of the damage-estimator encoder for the oracle gate head (a dummy map stands in for the GT map). Peak memory = MPS driver allocation (includes the allocator cache; measured process-wide, models profiled sequentially).
