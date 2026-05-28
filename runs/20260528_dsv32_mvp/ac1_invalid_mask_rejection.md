# AC-1 negative test: invalid CHANNEL_MASK_PATH → validator rejection (fail-closed)
# boot: CHANNEL_MASK_PATH=/models/DOES_NOT_EXIST_dsv32_mask.safetensors, port 30011, 2026-05-28 22:49:58
# Expected: validator raises at startup before model load; no silent dense fallback.

=== verbatim traceback tail from boot log ===
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/sgl-workspace/sglang/python/sglang/launch_server.py", line 69, in <module>
  File "/sgl-workspace/sglang/python/sglang/launch_server.py", line 50, in run_server
  File "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py", line 2353, in launch_server
  File "/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py", line 762, in _launch_subprocesses
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 7198, in check_server_args
    validate_double_sparsity(self)
  File "/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/validator.py", line 192, in validate_double_sparsity
    mask = load_channel_mask(config.channel_mask_path)
  File "/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/channel_mask.py", line 140, in load_channel_mask
    raise DoubleSparsityChannelMaskMissing(
sglang.srt.layers.attention.double_sparsity.channel_mask.DoubleSparsityChannelMaskMissing: channel mask file not found at '/models/DOES_NOT_EXIST_dsv32_mask.safetensors'. Set 'channel_mask_path' in --double-sparsity-config to a readable file.
