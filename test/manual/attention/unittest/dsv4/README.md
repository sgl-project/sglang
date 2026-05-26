# DSV4 Attention Capability Matrix

This folder is reserved for DeepSeek-V4 style attention tests. DSV4 has
method-specific sparse/indexer metadata, so it should not be folded into the
dense or MLA folders.

## Planned Matrix

| Backend family | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| DSV4/DeepSeek-V4 attention backends | Not implemented | Not implemented | Not implemented | Needs a small DSV4-specific fixture and independent reference. |

## Required Fixture Work

- Model the DSV4 indexer/KV-cache layout directly instead of reusing dense Q/K/V helpers.
- Add a PyTorch reference for the selected sparse/indexed attention behavior.
- Identify local hardware gates for B200/GB300-specific paths before adding default-discovery tests.

## First Test Target

- `dsv4/test_<backend_name>.py` for a locally runnable backend or a hardware-gated backend file with clear skips.
