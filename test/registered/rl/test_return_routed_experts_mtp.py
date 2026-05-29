"""End-to-end regression for `--enable-return-routed-experts` under
Frozen-KV MTP (the failing case fixed by the per-`TopKConfig`
`capture_routed_experts` opt-out).

The failure mode this guards against: draft / MTP MoE layers writing into
the target's R3 routed-experts device buffer, then `copy_to_cpu()` on the
overlap thread reading the (now polluted) view, then `Miles` receiving
draft-overwritten routing. The fix lives at construction time: every MoE
TopK in the draft NextN model carries `capture_routed_experts=False`, so
the CUDA-graph capture observes a Python constant `False` at record time
and the draft write kernel is never baked into the draft graph.

This file is a scaffold. The real test requires a multi-GPU host with the
target DeepSeek-V3-class MTP weights present; it is intentionally skipped
in environments that lack either. The assertions and launch shape are
documented inline so a CI configuration can drop in concrete weights and
arguments without further design work.
"""

import os
import unittest


# Real assertions require:
# 1. A target DeepSeek-V3-class model with NextN MTP weights cached locally.
# 2. >= 4 GPUs with enough HBM for tp=4 + speculative scratch.
# 3. SGLang server launch infra (mirrors `test_return_routed_experts.py`).
# Skip by default to avoid false positives in environments missing any of
# the above; CI workflow can flip _RUN_MTP_R3 to "1" when prerequisites are
# satisfied.
_RUN_MTP_R3 = os.environ.get("SGLANG_RUN_MTP_R3_REGRESSION") == "1"


@unittest.skipUnless(
    _RUN_MTP_R3,
    "Frozen-KV MTP R3 regression requires GPU + MTP weights + multi-GPU "
    "scratch; set SGLANG_RUN_MTP_R3_REGRESSION=1 to enable.",
)
class TestReturnRoutedExpertsFrozenKVMTP(unittest.TestCase):
    """Failing-case regression: Frozen-KV MTP + overlap + cuda-graph
    + --enable-return-routed-experts must produce target-equivalent
    routed-experts output.

    Test matrix (each variant must pass independently):
      - overlap=on, cuda-graph=on, piecewise=off (default overlap path)
      - overlap=on, cuda-graph=on, piecewise=on  (BYPASS path)
    Both compared against a target-only baseline (no spec decode, same
    prompt, same temperature=0/greedy, same seed/weights) for byte-equal
    routed_experts per layer.

    Implementation notes for CI wiring:
      - Mirror the server-launch infrastructure in
        ``test_return_routed_experts.py``: ``popen_launch_server`` plus
        the `_collect_results` helper, but launch with the MTP draft model
        path and ``--speculative-algorithm <eagle|nextn>`` selecting the
        Frozen-KV MTP runner.
      - For the BYPASS variant, add ``--enable-piecewise-cuda-graph``.
      - Per-layer per-token comparison must use
        ``extract_routed_experts_from_meta_info`` to decode the base64
        meta_info payload; iterate layers and assert cell-wise equality.

    The assertion contract:
      assert returned.shape == (num_hidden_layers, completion_tokens,
                                num_experts_per_tok)
      assert (returned == target_only_baseline).all()
    """

    def test_overlap_cuda_graph_default(self) -> None:
        self.skipTest("Pending CI wiring; see module docstring.")

    def test_overlap_cuda_graph_bypass_piecewise(self) -> None:
        self.skipTest("Pending CI wiring; see module docstring.")


if __name__ == "__main__":
    unittest.main()
