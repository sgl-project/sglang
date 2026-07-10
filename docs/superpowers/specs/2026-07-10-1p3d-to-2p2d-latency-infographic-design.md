# 1P3D to 2P2D Full-Chain Latency Infographic Design

## Objective

Create one landscape PNG infographic, visually modeled on the supplied D-to-P commit/abort reference, that accurately shows the executed path and measured latency of the `pd_flip_1p3d_to_2p2d_20260708_121443` run.

The figure must not present this run as a successful role flip: four KV entries transferred, target commit and source finish returned successfully, but the post-migration idle assertion timed out and the controller rolled the migration back. The final topology therefore remained `1P3D / SAFE` rather than reaching `2P2D`.

## Data Sources

- `experiments/pd_flip_todo_20260708_raw_bundle/pd_flip_1p3d_to_2p2d_20260708_121443/pd_state_machine_full_chain_latency.csv`
- `experiments/pd_flip_todo_20260708_raw_bundle/pd_flip_1p3d_to_2p2d_20260708_121443/01_1p3d_to_2p2d_two_phase/migration_link/controller_actions.csv`
- `experiments/pd_flip_todo_20260708_raw_bundle/pd_flip_1p3d_to_2p2d_20260708_121443/01_1p3d_to_2p2d_two_phase/migration_link/migration_status_samples.csv`
- `experiments/pd_flip_todo_20260708_raw_bundle/README.md`

## Visual Structure

Use a clean white 16:9 canvas with rounded black-outline process boxes, blue directional arrows, orange latency tags, and a red failure/rollback path. Use Chinese explanatory text with endpoint/action names retained in English where that improves precision.

The composition has three zones:

1. **Top summary:** title `1P3D → 2P2D：真实 KV 迁移全流程延迟`, badges for controller total `19.387 s`, controller migration `2.271 s`, and `4/4 KV entries transferred`.
2. **Executed main path:** a centered flow from router drain through KV transfer, commit, finish, and the failed idle assertion. This path is visually dominant.
3. **Outcome branches:** the red actual branch shows abort and cleanup returning to `1P3D / SAFE`; the unexecuted success tail is gray and labeled `本次未执行 / n/a`.

## Required Measured Labels

Display these values verbatim, rounded as shown:

| Stage | Label |
|---|---:|
| Router drain source | `0.34 ms` |
| Pause source admission | `5.47 ms` |
| Observe source quiesce | `15.020 s` |
| Scan running + waiting | `0.02 ms` |
| Build manifests | `0.22 ms` |
| Freeze waiting queue | `0.01 ms` |
| Start source migration API | `33.29 ms` |
| Target prepare receiver API | `15.91 ms` |
| KV transfer to target held | `1.912 s target / 1.806 s source` |
| Commit target | `8.65 ms` |
| Finish source | `13.36 ms` |
| Post-migration idle assertion | `2.009 s TIMEOUT` |
| Abort target | `13.50 ms` |
| Abort source | `6.87 ms` |
| Cleanup resume admission | `10.11 ms` |
| Cleanup router undrain | `0.41 ms` |

The gray unexecuted success tail contains:

- Set runtime role: `n/a`
- Refresh router role: `n/a`
- Normal resume admission: `n/a`
- Normal router undrain: `n/a`
- Intended terminal state: `2P2D / SAFE（未到达）`

## Accuracy Notes

- Label the 15.020-second quiesce value as a fixed observation window, not a measured drain-completion time.
- Do not show `Target held queue = 0 ms`; that CSV value is a state marker rather than measured residence latency.
- Describe the 1.912-second KV value as a worker-side transfer lifecycle proxy, not pure network latency.
- Do not add stage labels to obtain the 19.387-second total; the measurements mix controller wall time, worker-internal intervals, polling, and nested spans.
- State clearly that prefill/decode KV stitching was not exercised in this runner.

## Output

Generate a polished high-resolution PNG and save the final project artifact as:

`reports/pd_flip_1p3d_to_2p2d_full_chain_latency.png`

Preserve the reference image only as a style/layout reference; do not overwrite or edit it in place.
