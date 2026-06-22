# PD Runtime Role Flip Four-Node Docker Verification Results

Date: 2026-06-22

Branch: `pd-flip-execute-docker-ready`

## Local Verification

Passed:

```bash
python3 test/srt/test_pd_runtime_role_switch.py -v
python3 test/srt/test_pd_flip_active_decode_handoff.py -v
python3 test/srt/test_pd_flip_controller.py -v
python3 test/srt/test_pd_flip_experiment_script.py -v
python3 -m py_compile python/sglang/srt/managers/scheduler.py test/srt/test_pd_runtime_role_switch.py
python3 -m py_compile python/sglang/srt/managers/io_struct.py python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_active_decode_handoff.py
python3 -m py_compile scripts/playground/disaggregation/pd_flip_controller.py test/srt/test_pd_flip_controller.py
cd experimental/sgl-router && cargo test --lib
cd experimental/sgl-router && cargo test --test proxy pd_mode -- --test-threads=1
```

Observed results:

```text
test_pd_runtime_role_switch.py: 6 passed
test_pd_flip_active_decode_handoff.py: 4 passed
test_pd_flip_controller.py: 8 passed
test_pd_flip_experiment_script.py: 7 passed
sgl-router cargo test --lib: 409 passed
sgl-router proxy pd_mode subset: 8 passed
```

Additional execute-controller verification on this branch:

```bash
python3 test/srt/test_pd_flip_controller.py -v
python3 test/srt/test_pd_flip_experiment_script.py -v
python3 test/srt/test_pd_runtime_role_switch.py -v
python3 -m py_compile scripts/playground/disaggregation/pd_flip_controller.py test/srt/test_pd_flip_controller.py test/srt/test_pd_flip_experiment_script.py
```

Observed results:

```text
test_pd_flip_controller.py: 8 passed
test_pd_flip_experiment_script.py: 7 passed
test_pd_runtime_role_switch.py: 6 passed
py_compile: passed
```

Blocked in this local environment:

```bash
PYTHONPATH=python python3 test/srt/test_pd_flip_internal_state_update.py -v
PYTHONPATH=python python3 test/srt/test_pd_flip_state_machine.py -v
```

Both require the full SGLang Python dependency stack; this host currently fails
before test collection with:

```text
ModuleNotFoundError: No module named 'numpy'
```

`test/srt/test_disaggregation_fake_decode.py` is also not runnable on this host
because `torch` is not installed.

## Four-Node Docker Experiment

Not executed in this workspace. It requires four reachable 8-GPU servers, a
shared model path, Docker with GPU access, and the selected transfer backend
network configuration.

Use:

```bash
cd scripts/playground/disaggregation/pd_flip_docker
cp env.example env.local
export ENV_FILE=$PWD/env.local
```

Initial target layout:

```text
node0: prefill
node1: prefill
node2: decode
node3: decode
```

Run order:

```bash
# node0
./run_worker.sh prefill 0.0.0.0

# node1
./run_worker.sh prefill 0.0.0.0

# node2
./run_worker.sh decode 0.0.0.0

# node3
./run_worker.sh decode 0.0.0.0

# router host
./run_router.sh

# controller host
./run_controller.sh metrics
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute
```

The Docker defaults now map one physical 8-GPU node to one worker:

```text
TP_SIZE=8
DP_SIZE=1
TP_SIZE * DP_SIZE = 8
```

Use `DIRECTION=p_to_d SOURCE_NAME=node0 ./run_controller.sh execute` for the
reverse idle prefill-to-decode switch.

Fields to record during the real run:

```text
cluster nodes:
initial roles:
model:
transfer backend:
D->P source:
D->P target:
kv transfer seconds:
total flip seconds:
router unavailable errors:
request success count:
request failure count:
```

Controller result fields to save:

```text
success:
message:
source:
migration_target:
migration_seconds:
total_seconds:
actions:
```

Acceptance target:

```text
(router_refresh_complete - router_drain_start)
  <= 1.25 * (target_migration_complete - source_migration_start) + 2.0 seconds
```
