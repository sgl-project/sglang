# AMD MI35x cross-node PD-disagg CI bootstrap

Draft CI for **cross-node** 1P1D prefill/decode-disaggregated benchmarks on
AMD MI35x (gfx950) over the AMD **Pensando** RoCE fabric (8× `rdma0..rdma7`,
netdevs `tw-eth0..7`). Modeled after NV's PR sgl-project/sglang#22461 but
adapted for our infra (SSH + docker over Conductor nodes; later ARC on K8s).

> **Slurm caveat:** some nodes (e.g. the `mia1-p02-*` pool in `amd-frameworks`)
> are Slurm-gated by `pam_slurm_adopt` — SSH is refused until you hold an active
> allocation. Allocate first from a node that has the Slurm client:
> `salloc -A <account> -p <partition> -w <node> -t 02:00:00 --no-shell`, then SSH.
> Others (e.g. `mia1-p01-g20`, in the `rccl_dev` reservation) allow direct SSH.

## Layout

```
.github/workflows/nightly-pd-amd-crossnode.yml        workflow_dispatch + (commented) cron
ci_bootstrap/
├── README.md                                         (this file)
└── scripts/
    ├── preflight_nodes.sh     env / GPU / RDMA / container / port / IP-reach checks
    ├── setup_container.sh     start-if-stopped; fail-if-missing (no docker run yet)
    ├── discover_network.sh    read-only RDMA / netdev dump (pick PREFILL_IP/DECODE_IP)
    ├── run_cross_node_pd.sh   orchestrator: preflight -> setup -> launch -> bench -> collect
    ├── prefill_node.sh        runs inside container on prefill node, TP=8 all GPUs
    ├── decode_node.sh         runs inside container on decode node,  TP=8 all GPUs
    └── proxy_node.sh          sglang_router PD-disagg proxy
```

Orchestrator stages (run_cross_node_pd.sh):
**0** preflight  → **1** setup containers → **2** push role scripts →
**3** launch prefill+decode → **4** wait ports → **5** launch router →
**6** run cache_bench.py → **7** collect CSV+logs → **8** teardown

## How it differs from the in-repo 1p1d_bnxt_tp4_mtp/ launchers

The existing `1p1d_bnxt_tp4_mtp/{prefill,decode}_mori_nodp.sh` scripts are
**co-located 1P1D** on a single 8-GPU box: prefill grabs GPUs 0-3, decode
grabs 4-7, and the router talks to `0.0.0.0:30025` and `0.0.0.0:30100`.

The CI variants here put each role on **its own node** with all 8 GPUs:
- `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` everywhere
- `--host $HOST_IP` (RoCE/mgmt-routable, not 0.0.0.0) so the peer node can reach it
- RDMA devices (`IB_DEVS`) are **auto-discovered** from `ibv_devices` (the rdma*
  list), overridable via env; `SOCKET_IFNAME` defaults to `eno0` for the gloo/NCCL
  TCP control plane
- Router takes `PREFILL_IP` and `DECODE_IP` as env vars and points at remote URLs

## KV transfer backend (`mori` | `mooncake`)

Set `TRANSFER_BACKEND` (env or the `transfer_backend` workflow input) to pick the
disagg KV path. Default is `mori`.

- `mori` — AMD's native engine, validated path. The `MORI_IO_*` / `SGLANG_MORI_*`
  QP-tuning vars in the role scripts only apply here.
- `mooncake` — requires an image carrying the **#27730** mooncake bump
  (`MOONCAKE_COMMIT=d8f35569`, built with the multi-protocol HIP cmake flags).
  That bump pulls in **#2346**, the fix for the `deconstruct()` wild-pointer →
  `ibv_destroy_qp` segfault. Older images ship mooncake `v0.3.7.post2` (e.g. the
  `...20260623` mi35x image = `b6a841dc`) which **segfaults at QP teardown on
  disconnect** — runs may complete but the segfault is still in the logs and the
  path is fragile. Do not run the mooncake leg on a pre-#27730 image.

```bash
export TRANSFER_BACKEND=mooncake   # rest of the invocation is identical
bash ci_bootstrap/scripts/run_cross_node_pd.sh
```

## Two-phase rollout

### Phase 1 — manual trigger (now)

Run from your jumpbox or any host that has SSH keys for the Conductor nodes
and IP reachability to them. Either:

**(a) From the GitHub UI** once this PR is up: Actions → "Nightly PD-Disagg
Cross-Node Bench (AMD MI300X)" → Run workflow → fill the inputs. The
`runs-on: [self-hosted, arc-mi300x-pd]` label needs to resolve to a runner
you've registered by hand (any laptop with `./run.sh` works for a smoke
test, since the heavy lifting happens over SSH).

**(b) Locally without GitHub** — same script:

```bash
# Example: g20 (prefill, direct SSH) + g09 (decode, allocate via Slurm first).
# Use the eno0 mgmt IPs (10.24.112.x) — the addresses the jump server resolves.
export PREFILL_NODE=mia1-p01-g20
export DECODE_NODE=mia1-p02-g09
export PREFILL_IP=10.24.112.111       # g20 eno0
export DECODE_IP=10.24.112.140        # g09 eno0
export CONTAINER=mori_bench
export REMOTE_WORKDIR=/sgl-workspace/ci_pd
export MODEL_PATH=/data/models/DeepSeek-V4-Flash-MXFP4
bash ci_bootstrap/scripts/run_cross_node_pd.sh
```

Result goes to `./results-<timestamp>/` (CSV + per-role logs).

### Phase 2 — switch to ARC self-hosted runner (later)

Convert one Conductor MI35x node into a K8s worker, join yctseng's ARC
cluster, create a `RunnerScaleSet` named `arc-mi300x-pd`. Nothing in this
workflow needs to change — the `runs-on:` label already matches.

## Recommended order on a fresh pair of nodes

```bash
# 0) allocate any Slurm-gated decode node first (skip for direct-SSH nodes):
#    ssh mia1-p01-g20 'salloc -A amd-frameworks -p amd-frameworks -w mia1-p02-g09 -t 02:00:00 --no-shell'

# 1) dump per-node RDMA/netdev info (read-only) to confirm the rdma* fabric:
NODES="mia1-p01-g20 mia1-p02-g09" \
    bash ci_bootstrap/scripts/discover_network.sh | tee net-discovery.log

# 2) run preflight (IB_DEVS auto-discovers; set it only to pin specific NICs):
export PREFILL_NODE=mia1-p01-g20 DECODE_NODE=mia1-p02-g09
export PREFILL_IP=10.24.112.111  DECODE_IP=10.24.112.140
export CONTAINER=mori_bench
export REMOTE_WORKDIR=/sgl-workspace/ci_pd
bash ci_bootstrap/scripts/preflight_nodes.sh

# 3) if preflight is clean, run the full orchestrator:
bash ci_bootstrap/scripts/run_cross_node_pd.sh
```

## Open assumptions preflight will check for you

1. SSH reachability of both nodes.
2. `rocm-smi` is runnable AND not in CPX mode (>8 GPU entries warns).
3. `/dev/kfd`, `/dev/dri/renderD*`, `/dev/infiniband` populated.
4. `docker` reachable.
5. `$CONTAINER` exists on both nodes (and `setup_container.sh` will start it
   if stopped; fail-loudly if missing).
6. `$REMOTE_WORKDIR` + `cache_bench.py` present inside the container.
7. Expected ports (30025/8998 on prefill, 30100/9001 on decode, 8000 on router)
   are free.
8. At least one `rdma*` device is present on the host (and, if `IB_DEVS` is set,
   each requested device exists).
9. Peer IP is pingable from each node.

If preflight fails: read the FAIL lines, fix them, rerun. Do NOT skip with
`SKIP_PREFLIGHT=1` on the first manual run.
