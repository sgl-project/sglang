# Troubleshooting — DeepSeek-V4-Flash on Ascend NPU

Symptom → most likely cause → first action. Ordered by "how often it ends a lab session early".

| # | Symptom | Likely cause | First action |
| --- | --- | --- | --- |
| 1 | `npu-smi: command not found` / no devices | not inside the NPU container, driver missing, or devices not passed (`docker run` without `--device=/dev/davinci*`) | `ls /dev/davinci*`; check `docker run` used the documented device mounts; ask TA for the intended `drun` line |
| 2 | `import torch_npu` fails, or `torch.npu.is_available()` is `False` | `torch_npu` missing or major.minor mismatch with `torch` | `pip show torch torch_npu \| grep -E 'Name\|Version'`; use the validated image instead of pip-installing on top |
| 3 | `ACL_ERROR_*` / `CANN not found` at model load | `set_env.sh` not sourced in this shell (new shell = new environment) | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` and the other three sources in `deploy/03_launch.sh` |
| 4 | startup OOM / HBM near 100% before the server is ready | `--mem-fraction-static` too high for the die count, or context cap too large, or checkpoint bigger than the slice | step `--mem-fraction-static` down 0.75 → 0.70 → 0.65, then cut `--context-length`; verify the slice passes `env/00_common.sh` |
| 5 | `Allocator`/HCCL hang at init with tp > 1 | wrong `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`, port clash, or a device is unhealthy | run `--tp-size 1` first to prove the model path; check `npu-smi info -t health`; set both socket env vars to the actual NIC |
| 6 | `--attention-backend ascend` errors on V4 | V4's hybrid CSA/HCA attention needs the dedicated backend | use `--attention-backend dsv4` (per the V4 tutorial) |
| 7 | `--quantization modelslim` rejected / weights mismatch | checkpoint is not the ModelSlim W8A8 variant, or config lacks the quant description | confirm `config.json` `model_type=deepseek_v4` and the ModelSlim metadata; re-fetch `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp` |
| 8 | chat completion returns a template error / 400 | V4 ships **no Jinja chat template** (uses the model repo's `encoding/` API) | use the raw `/generate` endpoint for the benchmark, or SGLang's dsv4 encoding path; record which endpoint the numbers came from |
| 9 | speculative decoding crashes or gives garbage | draft/verify mismatch, MTP head not loaded, or EAGLE parameters from a different hardware profile | first run without `--speculative-*` to get a baseline, then add the documented 2/1/3 EAGLE settings |
| 10 | server never prints "ready to roll" but the process is alive | 300 GB checkpoint loading on a cold page cache | wait; watch `npu-smi info` HBM and `df`; first load can take tens of minutes |
| 11 | MXFP8/FP4 ops missing | you are on A2/A3, which have no MXFP8/MXFP4 (950PR/DT only) | use the W8A8 checkpoint; if the slice is 4×A2, the model does not fit — see `env/00_common.sh` message |
| 12 | benchmark TTFT unrealistically low | prefix cache hit from a previous cell | `--flush-cache` between cells (the matrix does this); restart the server between independent comparisons |
| 13 | throughput far below expectation | wrong `--tp-size` (e.g. 4 on A3 instead of 8), graph capture disabled, or quant not applied | check the recorded launch command; compare against the sanity bands in the practice report |
| 14 | `pip install` inside the container broke the environment | the image's CANN/torch_npu/kernels are a locked set | revert; recreate the container from the image instead of patching it |

## Escalation script (30 s, say it exactly like this)

> 我在 `<950PR/DT 4卡 | A2 4卡>` 的容器里,镜像 `<tag>`。
> `npu-smi info` 看到 `<N>` 个 device,CANN `<版本>`,torch_npu `<版本>`,sglang `<版本>`。
> 我执行 `<命令>`,得到 `<完整错误第一行>`。
> 需要确认 `<具体是 tp-size / 量化格式 / 显存还是权重路径>`。

Writing down the version tuple + the first error line before asking is what turns a 20-minute round trip into
a 2-minute answer.
| 15 | `Unexpected UE8M0 scale shape: got (8,8,32), expected (64,32)` | DeepSeek-V4 grouped `wo_a` block scales are 3-D; the 950 MXFP8 layout routine validates before reshaping | flatten the scale to the 2-D block grid before the check (`scale.reshape(expected)` in `w8a8_mxfp8.py::_layout_npu_arch35_ue8m0_weights`) |
| 16 | `aclnnKvQuantSparseAttnSharedkvMetadata … not in libopapi.so` | the `custom_transformer` vendor dirs are not on the CANN op path (the container sources only the base toolkit) | `source /usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/bin/set_env.bash` and the `customize` one, then `export ASCEND_CUSTOM_OPP_PATH=<those vendors>:…` |
| 17 | crash inside `decode_cuda_graph_runner` / `npu_cudagraph_backend` at `deepseek_v4.py` | NPU graph capture for DSv4 hybrid attention not usable in this build | launch with `--disable-cuda-graph` (eager decode), and label the resulting numbers as eager mode |
| 18 | `administratively prohibited: platform authorization denied` from the jump proxy | platform token is time/connection limited | regenerate the SSH command in the console; do everything you need in as few sessions as possible |
| 19 | `ModuleNotFoundError: No module named 'tilelang'` in `kernels/ops/layernorm/mhc.py` | DSv4 mHC layernorm kernels try a lazy tilelang import that the image lacks | set `SGLANG_OPT_USE_TILELANG_MHC_PRE=False` and `SGLANG_OPT_USE_TILELANG_MHC_POST=False` (vendor-validated switches) |
| 20 | `Failed to build JIT module sgl_kernel_jit_dpsk_v4_hash_topk_act_sqrt_softplus_false` | DSv4 fused hash top-k kernel cannot be JIT-built in this image | set `SGLANG_OPT_USE_FUSED_HASH_TOPK=False`; optionally `pip install ninja` |
| 21 | jump proxy returns `administratively prohibited: platform authorization denied` mid-session | token is time/connection limited | regenerate the SSH command; batch work into few sessions; keep servers running under `nohup` |
| 19 | `[ERROR][CamMoeDispatchNormal] HCCL_BUFFSIZE is too SMALL` then `aclnnCamMoeDispatchNormal failed … Failed to execute tiling function` | DeepEP MoE dispatch/combine needs a bigger HCCL window than the default 200 MB (the log states the exact requirement, e.g. 214 MB) | `export HCCL_BUFFSIZE=1200` and `export DEEPEP_HCCL_BUFFSIZE=2048` before launch (values from the vendor-validated A3 configs) |
