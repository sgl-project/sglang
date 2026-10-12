# Focused NPU offload / graph diagnostics

These are existing diagnostic probes packaged for review, not new full-model tests or automatic CI entries. No model weights, request audio, host identity or deployment paths are included.

- `python probe_state_dict_metadata_regression.py` reproduces the plain-dict metadata regression and checks a fresh metadata-preserving OrderedDict, including CPU Dynamo eager and changed weights. This standalone diagnostic mirrors the state-map contract; it does not import the final Offloader or claim NPU compiler coverage.
- With the proposed release-v0.5.20 SGLang package and inspected TorchNPU 2.10 installed: `python probe_fia_delivery_graph.py --device npu:0 --output /tmp/fia_report.json`. This reuses the existing synthetic same-capture length 1 -> 2 -> 1 diagnostic and expects 2 -> 4 -> 2. The public copy changes only output/device arguments; it was syntax-checked locally, not rerun on NPU for submission.
- `fia_delivery_graph_report.json` is the preserved numeric result of the earlier run. Original records are retained in the project delivery archive.

The earlier full acceptance used Qwen3-ASR 0.6B with CPU offload 0/4, compile and both graph stages enabled: 20 requests per arm, matching text in all pairs. That evidence belongs to the archived measured release and companion kernel patch, not arbitrary main or other versions. These probes do not broaden acceptance to every model, GPU offloader, layout or TorchNPU private registry.
