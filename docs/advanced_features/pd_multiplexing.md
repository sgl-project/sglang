
# PD Multiplexing


## Server Arguments

| Argument                     | Type/Default            | Description                                              |
|-----------------------------|-------------------------|----------------------------------------------------------|
| `--enable-pdmux`            | flag; default: disabled | Enable PD-Multiplexing (PD running on greenctx stream).  |
| `--pdmux-config-path <path>`| string path; none       | Path to the PD-Multiplexing YAML config file.            |

### YAML Configuration

Example configuration for an H200 (132 SMs)

```yaml
# Number of SM groups to divide the GPU into.
# Includes two default groups:
#   - Group 0: all SMs for prefill
#   - Last group: all SMs for decode
# The number of manual divisions must be (sm_group_num - 2).
sm_group_num: 8

# Optional manual divisions of SMs.
# Each entry contains:
#   - prefill_sm: number of SMs allocated for prefill
#   - decode_sm: number of SMs allocated for decode
#   - decode_bs_threshold: minimum decode batch size to select this group
#
# The sum of `prefill_sm` and `decode_sm` must equal the total number of SMs.
# If provided, the number of entries must equal (sm_group_num - 2).
manual_divisions:
  - [112, 20, 1]
  - [104, 28, 5]
  - [96, 36, 10]
  - [80, 52, 15]
  - [64, 68, 20]
  - [56, 76, 25]

# Divisor for default stream index calculation.
# Used when manual_divisions are not provided.
# Formula:
#   stream_idx = max(
#       1,
#       min(sm_group_num - 2,
#           decode_bs * (sm_group_num - 2) // decode_bs_divisor
#       )
#   )
decode_bs_divisor: 36

# Maximum token budget for split_forward in the prefill stage.
# Determines how many layers are executed per split_forward.
# Formula:
#   forward_count = max(1, split_forward_token_budget // extend_num_tokens)
split_forward_token_budget: 65536
```
