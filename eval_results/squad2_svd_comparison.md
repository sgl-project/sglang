# SQuAD v2 SVD Compression Results

Model: Qwen/Qwen3.5-4B | n-shot: 0 | limit: 11000 | batch_size: 16 | max_gen_toks: 1024

## Results

| Metric | No SVD (baseline) | Prefix SVD only | Prefix + Mamba SVD |
|--------|-------------------|-----------------|---------------------|
| **exact** | 49.2909 | 49.2818 | 49.2636 |
| **f1** | 49.7608 | 49.7486 | 49.7227 |
| HasAns_exact | 4.0298 | 4.0298 | 3.9935 |
| HasAns_f1 | 4.9680 | 4.9618 | 4.9100 |
| NoAns_exact | 94.7004 | 94.6822 | 94.6822 |
| NoAns_f1 | 94.7004 | 94.6822 | 94.6822 |
| best_exact | 49.9182 | 49.9182 | 49.9182 |
| best_f1 | 50.3234 | 50.3022 | 50.2992 |

## Deltas from Baseline

| Metric | Prefix SVD only | Prefix + Mamba SVD |
|--------|----------------|--------------------|
| **exact** | -0.009 | -0.027 |
| **f1** | -0.012 | -0.038 |
| HasAns_exact | 0.000 | -0.036 |
| HasAns_f1 | -0.006 | -0.058 |
| NoAns_exact | -0.018 | -0.018 |
| NoAns_f1 | -0.018 | -0.018 |
