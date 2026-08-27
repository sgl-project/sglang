"""Whether a captured DFLASH verify graph carries the tree mask.

`resolve_dflash_verify_mask_policy` answers one question at cuda-graph capture time:
does `spec_info.custom_mask` get a buffer? The answer is frozen into the graph -- for
`TritonAttnBackend` it becomes the `USE_CUSTOM_MASK` `tl.constexpr`, so a wrong answer
cannot be corrected at replay. Getting it wrong on a tree is silent: the runtime still
builds and copies the mask every step, and the graph verifies a 29-node tree as a plain
causal sequence. That was measured as 8/8 diverged completions before the tree branch
below existed, which is why it is worth a test of its own.

The chain direction is equally load-bearing in the other direction: chain verify is
genuinely causal, and re-enabling the mask for it would give up the backend's cheaper
built-in path for nothing.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# `dflash_tree_verify_active` imports `get_spec` lazily inside the function body, so
# patching the definition site is what takes effect.
_GET_SPEC = "sglang.srt.runtime_context.get_spec"


def _spec(*, tree_width):
    return mock.patch(
        _GET_SPEC,
        return_value=SimpleNamespace(speculative_dflash_tree_width=tree_width),
    )


class TritonAttnBackend:
    """Name-matched stand-in: the policy keys on the resolved class name, and importing
    the real backend drags in triton plus a device."""

    full_attn_backend = None


class AiterAttnBackend:
    """Any backend outside the chain skip set -- it has no built-in causal verify."""

    full_attn_backend = None


class _Hybrid:
    """Shape of `HybridLinearAttnBackend`: the tree's mask channel is the full-attention
    child, not the composite the model runner holds."""

    def __init__(self, child):
        self.full_attn_backend = child


class TestChainSkipsTheMask(CustomTestCase):
    def test_backend_with_a_builtin_causal_path_skips_the_mask(self):
        with _spec(tree_width=1):
            name, build_custom_mask = resolve_dflash_verify_mask_policy(
                TritonAttnBackend()
            )

        self.assertEqual(name, "TritonAttnBackend")
        self.assertFalse(build_custom_mask)

    def test_backend_without_one_still_gets_the_mask(self):
        with _spec(tree_width=1):
            _, build_custom_mask = resolve_dflash_verify_mask_policy(AiterAttnBackend())

        self.assertTrue(build_custom_mask)


class TestTreeAlwaysTakesTheMask(CustomTestCase):
    def test_skip_set_does_not_apply_to_a_tree(self):
        # The regression: siblings must not see each other, and only the mask says so.
        with _spec(tree_width=4):
            name, build_custom_mask = resolve_dflash_verify_mask_policy(
                TritonAttnBackend()
            )

        self.assertEqual(name, "TritonAttnBackend")
        self.assertTrue(build_custom_mask)

    def test_tree_width_one_is_still_a_chain(self):
        # Width 1 is the equivalence gate for every phase of this feature: it must keep
        # resolving byte-identically to the pre-tree behaviour.
        with _spec(tree_width=1):
            _, build_custom_mask = resolve_dflash_verify_mask_policy(TritonAttnBackend())

        self.assertFalse(build_custom_mask)


class TestCompositeBackendIsUnwrapped(CustomTestCase):
    def test_hybrid_target_resolves_to_its_full_attention_child(self):
        # The bench target (Qwen3.8 GDN + GQA) hands the runner a composite. Answering
        # for the composite's own class name would miss the skip set both ways.
        backend = _Hybrid(_Hybrid(TritonAttnBackend()))

        with _spec(tree_width=4):
            name, build_custom_mask = resolve_dflash_verify_mask_policy(backend)

        self.assertEqual(name, "TritonAttnBackend")
        self.assertTrue(build_custom_mask)


class TestTreeKeepsTheSelectorEager(CustomTestCase):
    """The other half of getting a tree into the graph: the draft head must stay out.

    Folding it in would give the beam one sampled path instead of the `[bs, gamma, K, K]`
    transition lattice it walks. This used to be a per-step `RuntimeError`; the graph is
    on by default now, so it has to be decided once, before capture.
    """

    def test_tree_verify_refuses_to_fold_the_draft_head(self):
        worker = SimpleNamespace(
            _use_tree_verify=True,
            ps=SimpleNamespace(tp_rank=0),
            block_size=8,
        )

        with mock.patch.dict(
            "os.environ", {"SGLANG_DFLASH_EAGER_DRAFT_SAMPLER": "0"}, clear=False
        ):
            self.assertIsNone(DFlashWorkerV2._maybe_build_draft_sampler(worker))

    def test_chain_still_reaches_the_folding_checks(self):
        # Guards the placement of the tree branch: if it were unconditional, the chain
        # would lose the folded head too. Reaching the target-lm_head lookup (and failing
        # on this stub) is the proof that it did not short-circuit.
        worker = SimpleNamespace(
            _use_tree_verify=False,
            ps=SimpleNamespace(tp_rank=0),
            block_size=8,
        )

        with mock.patch.dict(
            "os.environ", {"SGLANG_DFLASH_EAGER_DRAFT_SAMPLER": "0"}, clear=False
        ):
            with self.assertRaises(AttributeError):
                DFlashWorkerV2._maybe_build_draft_sampler(worker)


if __name__ == "__main__":
    unittest.main()
