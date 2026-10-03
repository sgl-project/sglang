# GLM-5.3-Flash MTP with PDMux

This is the fourth change in the GLM PDMux rollout, on
`feat/glm53-pdmux-mtp`. It depends on model/scheduler support,
Mamba/HiCache state handling, and DP/EP support, in that order.
The original experiment remains on `feat/glm53-flash-pdmux`.
DSpark was reverted in favor of the checkpoint's single-layer MTP block.

GLM's checkpoint NextN/MTP block runs through single-layer `EAGLE` (`NEXTN`
is an alias). PDMux previously called a missing split-prefill method on
`EAGLEWorkerV2`, and eager verify reused the in-flight prefill attention backend.

The target now captures all token hidden states across prefill slices. After
the last slice, MTP draft extend rotates the original prompt tokens and seeds
the next speculative round. It uses a separate batch view, including on idle
DP ranks, so target inputs and split progress survive unchanged. The draft
uses the prefill TP communicator and waits for outstanding decode work before
reusing its planner and index-share buffers. Earlier target slices can overlap
decode as before.

Verify masks, eager verify/idle attention, and accepted Mamba states use the
selected decode backend. GLM target and NextN helper streams are permitted
only on full-device decode lanes. Draft decode/extend graphs and draft prefill
graphs remain disabled under PDMux because their captures do not follow its
stream partitions; MTP draft forwards run eager. Target decode/verify graphs
retain the existing per-stream capture path. Split-prefill idle ranks always
execute their requested layer interval rather than replaying a decode graph.
Speculative planner work runs on the selected compute stream under PDMux.

## Launch

Keep the existing GLM PDMux launch and SM allocation. Replace any DSpark flags
with these arguments, using the **same GLM-5.3-Flash checkpoint** for target and
draft so its NextN weights are loaded:

```text
--speculative-algorithm EAGLE
--speculative-draft-model-path /path/to/GLM-5.3-Flash
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6
```

The target and draft retain the GLM DSA backend. Remove any DSpark-specific
`--speculative-draft-attention-backend flashinfer` override. As before, PDMux
requires `--disable-overlap-schedule`, PP1, no mixed chunk and no PD
disaggregation. Multi-layer EAGLE, adaptive speculative parameters, EAGLE3,
DSpark and other speculative algorithms are rejected with PDMux.

## Validation

Run the focused tests in a complete SGLang serving environment:

```bash
PYTHONPATH=python python3 -m unittest discover \
  -s test/registered/unit/spec -p test_eagle_pdmux.py -v
PYTHONPATH=python python3 -m unittest discover \
  -s test/registered/unit/models -p test_glm5_next_pdmux.py -v
```

The reorganized stack was rebased onto upstream `main` at `5b5d721239`.
On macOS with Python 3.12 and real CPU PyTorch, the focused suite ran 179 tests:
177 passed and two were skipped. This includes all 14 tests in
`test_eagle_pdmux.py`, plus GLM split-forward equivalence with the upstream
batch-owned residual interface, DP padding, Mamba reservations, overlap streams,
and HiCache event handling. Kernel launches and collectives are mocked where
these CPU tests require CUDA.

The larger unified radix-cache matrix cannot complete on this host: its native
HiCache hash requires little-endian Linux, and CUDA fixtures are unavailable.
CUDA kernels, target graph replay, eight-rank stress, GPU accuracy and MTP
throughput remain unvalidated. On the GPU host, compare with MTP enabled and
PDMux disabled, then exercise concurrent long prefills and decode, uneven DP
ranks, chunked prefill and HiCache. Repeat with target CUDA graphs disabled to
isolate graph-specific issues.
