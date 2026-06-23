# MI355 multi-node PD-disaggregation nightly

AMD MI355 parity for the GB200 nightly perf-regression pipeline
(`.github/workflows/nightly-72-gpu-gb200.yml`, NVIDIA srt-slurm). Tracks
multi-node SGLang **prefill/decode disaggregation** (mooncake transfer backend,
AITER attention) on the `amd-sglang` Slurm partition. See issue #22464.

## Layout

| File | Role |
| --- | --- |
| `.github/workflows/nightly-2-node-mi355.yml` | Workflow: matrix → one Slurm job per recipe → artifacts + summary. |
| `scripts/ci/slurm/nightly-configs-mi355.yaml` | Declarative config list (`--runner mi355`). |
| `scripts/ci/slurm/launch_mi355.sh` | Reads a recipe, submits `pd_disagg.slurm`, streams the log, collects result JSONs. |
| `scripts/ci/slurm/mi355/pd_disagg.slurm` | Allocates `prefill_nodes + decode_nodes` MI355 nodes; one Docker container per node. |
| `scripts/ci/slurm/mi355/server_pd.sh` | Runs in-container; branches prefill/decode by node rank; rank 0 runs router + benchmark. |
| `scripts/ci/slurm/mi355/recipes/*.yaml` | Topology + server flags (schema shared with `process_result.py`). |
| `scripts/ci/slurm/generate_matrix.py` | Reused unchanged (generic over `--runner`). |
| `scripts/ci/slurm/process_result.py` | Reused; `HW=mi355` selects the hardware label. |
| `scripts/ci/slurm/summarize.py` | Reused; title derived from the `hw` field. |

## Why Docker (not pyxis/enroot)

The `amd-sglang` partition has no pyxis SPANK plugin, so containers are launched
with `docker run` inside the allocation (host network/IPC, full ROCm device set:
`/dev/kfd`, `/dev/dri`, `/dev/infiniband`; `--privileged`; `--shm-size 128G`).

## Local test (no runner needed)

```bash
# From a sglang checkout on the login node (mia1-vm-amd-prj3-k8s-005):
export GITHUB_WORKSPACE=$PWD RUNNER_NAME=local-test
export FRAMEWORK=sglang-disagg MODEL_PREFIX=qwen3-8b PRECISION=bf16
export ISL=1024 OSL=1024
export CONFIG_FILE=scripts/ci/slurm/mi355/recipes/1p1d-tp8.yaml
export RESULT_FILENAME=mi355-qwen3-8b
export IMAGE=<rocm/sgl-dev:...-mi35x...>     # mooncake + AITER image
export NODELIST=mia1-p02-g09,mia1-p02-g29    # optional: pin prefill,decode
bash scripts/ci/slurm/launch_mi355.sh
```

Default ports: prefill server `30000` (bootstrap `8998`), decode server `30001`
(bootstrap `9001`), router `8000`. Override via `*_PORT` env vars.
