# Running an SGLang CI runner from the CI image

`lmsysorg/sglang:dev-ci` is a complete self-hosted runner for `sgl-project/sglang`:
the GitHub Actions runner plus the environment our CUDA CI jobs install, built nightly
from `docker/ci.Dockerfile`. A GPU host needs Docker, the NVIDIA Container Toolkit and
one `docker run` per runner.

## Scope

- **Blackwell GPUs only** (B200, B300, GB300). The image is pre-installed for SM100+;
  the health check refuses other GPUs, which would reinstall dependencies on every job.
- **Test labels only.** The image has no Docker CLI, so it cannot serve the labels whose
  jobs build kernels or wheels with `docker run`.
- **CUDA:** the image installs the same `CU_VERSION` as the rest of the fleet (`cu130`
  today). Bumping it is a one-line change to the `CU_VERSION` build argument.

## What a maintainer provides

- A **registration token**, from `gh api -X POST repos/sgl-project/sglang/actions/runners/registration-token`.
  It expires after an hour and is needed only for a runner's first start.
- The **runner name** and **label** for each runner (below).

## Starting runners

Run one container per runner. Split the host's GPUs between them with `--gpus`, and give
each runner its own volume; all runners on a host share one model volume.

- `RUNNER_NAME` (required): `<org>-<gpu>-<host>-<gpu ids>`, lowercase, e.g. `acme-gb300-host01-0123`.
- `RUNNER_LABELS` (required): one `<count>-gpu-<gpu>` label, e.g. `4-gpu-gb300`. Its count
  must match the GPUs given to the container.
- `RUNNER_TOKEN`: first start only.
- `PREFETCH_MODELS` (optional): space-separated Hugging Face repo IDs to download before
  taking jobs; otherwise each model downloads the first time a test needs it.
- `HF_TOKEN` (optional): for gated models. Jobs can read it, so use a read-only token.

An 8-GPU GB300 host as two 4-GPU runners:

```bash
for slot in 0123:0,1,2,3 4567:4,5,6,7; do
  name="acme-gb300-host01-${slot%%:*}"
  docker run -d --restart unless-stopped --name "${name}" \
    --gpus "\"device=${slot#*:}\"" \
    --ipc=host --cap-add SYS_PTRACE \
    -e RUNNER_NAME="${name}" \
    -e RUNNER_LABELS=4-gpu-gb300 \
    -e RUNNER_TOKEN=<registration token> \
    -v sglang-ci-models:/opt/sglang-ci-models \
    -v "${name}:/root/.cache" \
    lmsysorg/sglang:dev-ci
done
```

`SYS_PTRACE` is required: CUDA IPC and FlashInfer's MNNVL all-to-all share handles between
sibling processes, which the default ptrace scope blocks.

**GB300 multi-GPU (not yet verified).** Our GB300 setup suggests these are also needed:
`--device` for each `/dev/nvidia-caps-imex-channels/channel*` (MNNVL all-to-all),
`--ulimit memlock=-1`, and for DeepEP `--network host` and `--device /dev/infiniband`.

**Process visibility.** Without `--pid=host`, `nvidia-smi` in the container may not show
GPU process IDs. The health check judges idleness by GPU memory, so it still works; add
`--pid=host` if you want leftover processes from outside the container cleaned up too.

## Health checks

A runner takes jobs only while its GPUs are healthy. The check confirms the label's GPU
count is visible, every GPU is Blackwell and idle (under 2 GiB in use), a small matmul
succeeds on each GPU, and both volumes have at least 50 GB free (`MIN_FREE_DISK_GB`).

- **At start**, the container retries the check every minute and does not register or
  take jobs until it passes.
- **Before each job**, the runner kills leftover SGLang processes and reruns the check. If
  it fails, that job fails and the runner goes offline until the check passes again.
- **After each job**, leftover SGLang processes are killed.

## From smoke to production

A new runner registers only with `smoke-<label>`, e.g. `smoke-4-gpu-gb300`, so it never
takes real jobs untested. After a smoke run on that label passes, a maintainer replaces
its labels:

```bash
gh api -X PUT repos/sgl-project/sglang/actions/runners/<runner id>/labels -f 'labels[]=4-gpu-gb300'
```

The change is stored with the registration, so it survives restarts and upgrades.

## Upgrading

`--restart` never pulls a new image. To move to the latest nightly, recreate each container
with the same command, minus `RUNNER_TOKEN`:

```bash
docker pull lmsysorg/sglang:dev-ci
docker rm -f acme-gb300-host01-0123
docker run ...   # as above, without RUNNER_TOKEN
```

The registration and caches live in the volumes, so no new token is needed.

## Disk

Nothing evicts models or compiled kernels; the volumes grow with every model the runners
use. Plan on roughly 100 GB per host for the 4-GPU suites, and much more for nightlies. When
a runner refuses jobs for low disk, remove unused models with `hf cache delete` inside a
container, or drop a runner's volume while its container is stopped.

## Not covered yet

- **Monitoring:** partner hosts send no logs, metrics or heartbeat; a bad runner shows up
  only as failing jobs.
- **Removing a runner:** a maintainer deletes it in the repository's runner settings, then
  the owner removes its container and volume.
