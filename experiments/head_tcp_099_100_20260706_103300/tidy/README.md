# Tidy raw data

These files are easier-to-read views generated from the original raw experiment output.

- `summary_by_size.csv`: one row per transfer size per rate. This is the main table for plots and comparisons.
- `samples_long.csv`: one row per measured repeat. This is the real per-run raw timing data; each size has 20 rows per rate.
- `target_info.csv`: target-side registration/session metadata copied from cloud-100.

Original raw files are still under `source/` and `target/`.
