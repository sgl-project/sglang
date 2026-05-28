Run failed. No `/models/dsv32-fp8-channel-mask.safetensors` artifact was produced.

Wall-clock:
- Exact command reached `deepseek_v32` config loading failure in `8s`.
- After diagnostic fixes to get past dispatch/dependency blockers, checkpoint loading failed in `18s` with CUDA OOM. Log: [dsv32-fp8-channel-mask.calibrate.log](/tmp/dsv32-fp8-channel-mask.calibrate.log)

Validation output:

```text
DoubleSparsityChannelMaskMissing: channel mask file not found at '/models/dsv32-fp8-channel-mask.safetensors'. Set 'channel_mask_path' in --double-sparsity-config to a readable file.
```

Calibration warnings/diagnostics:
- `torch_dtype` is deprecated; Transformers says to use `dtype`.
- Hugging Face load report showed many `UNEXPECTED` keys, including expert weights/scales and `self_attn.indexer.*`.
- Load report showed `MISSING` fused MoE params: `mlp.experts.down_proj`, `gate_up_proj`, and scale tensors for layers `{3...60}`.
- Final failure: CUDA OOM on GPU 0, trying to allocate `3.50 GiB` with only `683 MiB` free while the process held about `139.12 GiB`.

I reverted the temporary source edit used for diagnosis; the git worktree is clean. I did install `accelerate==1.13.0` while checking the runtime blockers.
