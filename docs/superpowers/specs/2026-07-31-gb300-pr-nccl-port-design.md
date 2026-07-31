# GB300 PR Test NCCL Port Design

## Goal

Prevent Torch TCPStore rendezvous failures caused by dynamically selected
ephemeral NCCL ports in GB300 pull-request tests, without changing the global
`get_free_port()` implementation.

## Scope

Apply explicit NCCL ports to every SGLang server launch in PR tests registered
with `runner_config="4-gpu-gb300"`:

- `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`: two launches
- `test/registered/ep/test_flashinfer_a2a.py`: three launches
- `test/registered/disaggregation/test_disaggregation_aarch64.py`: two
  concurrent launches

Do not change:

- GB300 nightly tests under `test/registered/gb300/`
- `test_numa_utils.py`, which does not launch a distributed server
- `test_flashinfer_comm_fusion.py`, which uses mocked communication objects
- production port allocation or `get_free_port()`

## Port Allocation

Derive all ports from `DEFAULT_PORT_FOR_SRT_TEST_RUNNER` and reserve a distinct
block per PR test file:

| Test file | Offsets | GB300 CI ports |
| --- | --- | --- |
| DeepSeek CuteDSL | `+100`, `+101` | `10100`, `10101` |
| FlashInfer A2A | `+110`, `+111`, `+112` | `10110`–`10112` |
| Disaggregation AArch64 | `+120`, `+121` | `10120`, `10121` |

These values remain below the GB300 runner's ephemeral range beginning at
`10240`. Distinct ports prevent overlap between sequential server launches and
between the concurrently running prefill and decode servers.

Each affected launch passes its assigned value through `--nccl-port`. The
existing DeepSeek assignments remain unchanged.

## Failure Handling

This change only removes dynamic NCCL rendezvous port selection from the
affected tests. Server launch, cleanup, offline retry, and test failure behavior
remain unchanged. Any subsequent non-port failure is reported by the existing
test harness.

## Verification

Before implementation, run a static regression check and confirm that it fails
because FlashInfer A2A and disaggregation launches lack explicit NCCL ports.
After implementation:

1. Run the same static check to confirm all seven launches have distinct ports.
2. Run Ruff formatting and lint checks on affected files.
3. Compile the affected Python files.
4. Run repository pre-commit checks, including CI registration validation.
5. Push the update to PR #33044.
6. Trigger the same four-file `/rerun-test` command used previously and monitor
   every resulting workflow to completion.

The full inference validation requires the remote 4×GB300 runner.
