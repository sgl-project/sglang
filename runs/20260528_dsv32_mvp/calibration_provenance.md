# DeepSeek-V3.2 channel-mask calibration provenance (Round 1)

Date: 2026-05-28
Host: 8× H200 (143 GB free each), single node.
Code: commit `c99ed3644` (calibrate.py V3.2 loader redesign).

## Output mask
- Path: `/models/dsv32-fp8-channel-mask.safetensors` (999,992 bytes)
- `content_sha256 = 7b3207cae888c141173703384bfd7c8974b7adb64b1fddbdacac3ab26c7d6ac6`
- Metadata (via `load_channel_mask`): `dtype=fp8_e4m3`, `page_size=64`, `label_dim=16`,
  `head_dim=128`, `channel_selection` = `int32 (61, 128, 16)`, channel indices in `[0, 128)`.
- Validation output: `mask_validation.txt`.

## Load path (Blocking Side Issues #1/#3, resolved)
- HF `AutoModelForCausalLM` cannot load `deepseek_v32`. calibrate.py remaps the config
  `model_type` → `deepseek_v3` (`architectures=[DeepseekV3ForCausalLM]`) and loads the FP8
  weights under the transformers V3 MLA modeling. V3.2 = V3 + the DSA indexer, which is
  irrelevant to channel-importance calibration (only `kv_b_proj`/`q_b_proj` matter).
- DeepGEMM hub kernel is skipped (`_force_triton_fp8_for_calibration`); the forward uses
  transformers' `finegrained-fp8` Triton kernel. Run online (NOT `HF_HUB_OFFLINE`) so the
  kernel's publisher-trust check passes.

## Corpus (Queued issue, resolved)
- Pile-val (`mit-han-lab/pile-val-backup`) is zstd-compressed; `pip install zstandard`
  enables `datasets` to read it.
- Local corpus built by streaming Pile-val and keeping the first 300 docs with
  `len(text.strip()) >= 1500`, one doc per line (internal whitespace collapsed):
  `runs/20260528_dsv32_mvp/calib_corpus_pileval.txt`
  (`sha256 = 46d72075ed1d64831cf6...`, 300 lines, 3.6 MB). Gitignored for size; rebuildable
  from the streaming filter above. (Pile-val streaming order is not seed-deterministic, so
  the corpus is committed-by-reference; the mask's provenance is this log + the content SHA +
  the later AC-1.1 genuine-sparsity check.)

## Exact command
```
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
    --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --tp 8 \
    --output /models/dsv32-fp8-channel-mask.safetensors \
    --label-dim 16 --page-size 64 --num-samples 256 --block-size 512 --seed 42 \
    --dataset runs/20260528_dsv32_mvp/calib_corpus_pileval.txt -v
```
Log: `calibrate_full_20260528-171953.log`. Wall-clock ~8 min (load ~5 min + 256 forwards).

## One-block dry-run (fail-closed gate, passed)
Same command with `--dry-run-blocks 1` and the small `dryrun_prompts.txt`: confirmed FP8
sharded across all 8 GPUs (no upcast), placement validator passed, and the Method-1 Q/K
hooks fired on all 61 layers before the full run. Log: `calibrate_dryrun5_20260528-171332.log`.
