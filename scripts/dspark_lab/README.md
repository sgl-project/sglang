# DSpark Resident GPU Lab

This worker runs one experiment at a time on one allocated GPU. A small,
interruptible BF16 matrix multiplication load runs while the experiment queue
is empty. The load is synthetic; it is not a training or performance result.

The worker never imports PyTorch. Before launching an experiment it terminates
and reaps the idle child, releasing its CUDA context. At experiment completion
it terminates descendants and restarts the idle load. Experiments must remain
in their process group; do not daemonize them or call `setsid` inside them.

The state directory must be on a shared filesystem visible at the same path on
the submission host and the GPU host. Only trusted users should be able to write
it: queue entries deliberately execute the submitted command with job privileges.

```bash
python3 scripts/dspark_lab/gpu_worker.py --state-dir /shared/dspark-lab/state serve
python3 scripts/dspark_lab/gpu_worker.py --state-dir /shared/dspark-lab/state status
python3 scripts/dspark_lab/gpu_worker.py --state-dir /shared/dspark-lab/state submit \
  --cwd /shared/dspark-lab/sglang --timeout-seconds 1800 -- python3 experiment.py
python3 scripts/dspark_lab/gpu_worker.py --state-dir /shared/dspark-lab/state pause
python3 scripts/dspark_lab/gpu_worker.py --state-dir /shared/dspark-lab/state resume
```

`submit` prints a job ID. Results are under `results/<id>.json`; stdout/stderr
are under `logs/<id>.log`. `pause` disables only the idle load, not submitted
experiments. An experiment timeout kills its process group. Interrupted tasks
are reported as interrupted and are not replayed automatically. A single-worker
filesystem lock prevents accidental concurrent consumers.

The default idle duty cycle is 60%, with three 8192-square BF16 matrices
(approximately 384 MiB plus the CUDA context and library workspace). The actual
GPU utilization can differ from the duty cycle. No GPU memory is deliberately
reserved for the idle load while an experiment runs.

Northjob may preempt/restart the allocation according to cluster policy. A
resident job is not a guarantee of uninterrupted physical-node ownership.
This worker does not change cluster priorities or reclaim policies.

## Allocated H100

The initial resident allocation is:

| Item | Value |
| --- | --- |
| Northjob display name | `dspark-h100-resident-20260930` |
| Job ID | `job-fe1ce1dcdea6-20261001023258` |
| Namespace | `qiji` |
| Pod | `job-fe1ce1dcdea6-20261001023258-master-0` |
| Container | `master` |
| Initial node | `node064` |
| GPU | One NVIDIA H100 80GB HBM3 |
| Host memory allocation | 120 GiB |
| Image | `harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15` |
| Shared lab directory | `/gpfs/users/fuxuanwei-1/dspark-maas-lab` |

The image reports PyTorch `2.11.0+cu130`, CUDA `13.0`, SGLang `0.5.15`,
and Transformers `5.12.1`. The installed SGLang is not the implementation
checkout: experiments must explicitly select the code revision they exercise.
The image also contains the Mooncake Python package and master/client binaries.
Local model weights include `/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B`.

The submission host's `/gpfs/user/fuxuanwei/sglang` is not mounted in the pod.
Stage experiment code under the shared lab directory and use that path as
`--cwd`. Mounts such as `/gpfs/models` are only available inside the pod.
Do not overwrite experiment source files while a submitted task is using them.

Run these commands on the submission host:

```bash
LAB=/gpfs/users/fuxuanwei-1/dspark-maas-lab

# Status includes the queue length and heartbeat age. A stale heartbeat means
# the saved mode does not prove the worker is still alive.
python3 "$LAB/bin/gpu_worker.py" --state-dir "$LAB/state" status

# Commands run inside the GPU container. The returned ID identifies result/log files.
python3 "$LAB/bin/gpu_worker.py" --state-dir "$LAB/state" submit \
  --cwd "$LAB" --timeout-seconds 120 -- nvidia-smi

# Pause only the background load; experiments continue to be accepted.
python3 "$LAB/bin/gpu_worker.py" --state-dir "$LAB/state" pause
python3 "$LAB/bin/gpu_worker.py" --state-dir "$LAB/state" resume

kubectl get pod -n qiji job-fe1ce1dcdea6-20261001023258-master-0 -o wide
```

Idle output is in `$LAB/state/logs/idle.log`. Each experiment produces
`$LAB/state/results/<job-id>.json` and `$LAB/state/logs/<job-id>.log`.
Container restarts load the scripts from `$LAB/bin`; updating those files does
not hot-reload an already running worker. The state directories and atomic JSON
outputs retain the shared directory owner's UID/GID when the container runs as
root, so the submission user can continue enqueueing tasks.

To release the allocation when it is no longer needed (also stops experiments):

```bash
northjob delete job-fe1ce1dcdea6-20261001023258 --namespace qiji
```

## Validation

The initial CUDA smoke task, `01790793452910116102-4749255f56cd`, completed
successfully. It checked device availability, runtime versions, and a finite
CUDA matrix multiplication. The idle process exited before the experiment and
restarted with a new PID afterwards. Observed idle usage was approximately
1.1 GiB of device memory and 59-62% GPU utilization with the default settings.

Two additional queued tasks checked failure recovery inside the resident pod:
`01790794183509523367-4cd59b0fe386` exited with code 7 and was recorded as
`failed`; `01790794183585010218-fd62276053ad` exceeded its one-second limit
and was recorded as `timed_out` with code 124. The idle process then restarted,
changing from PID 151 to PID 249, with no active experiment remaining.

CPU process-lifecycle checks also exercised pause/resume, execution while
paused, timeouts, and worker shutdown. These checks validate the lab runner;
they do not establish SGLang capture correctness, DSpark training correctness,
or cross-machine RDMA performance. This single-GPU pod has no requested RDMA
device and is subject to the cluster's reclaim policy.
