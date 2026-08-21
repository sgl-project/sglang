"""ServerArgs-level tests for DFLASH tree drafting (phase A: config surface only).

What these lock down: DFLASH used to carry one width (`speculative_num_draft_tokens`) that
served as both the draft block width and the target's verify width. Tree drafting splits it
into three -- block width, verify width `1 + (block_size - 1) * tree_width`, and the per-depth
beam width parked in `speculative_eagle_topk`. Tree width 1 must resolve byte-identically to
the pre-split behaviour, since that equality is the regression gate for every later phase.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_HOOK = "sglang.srt.arg_groups.speculative_hook"


def _draft_config(block_size=8, selector_rank=64, selector_top_k=16):
    """Stand-in for the parsed `dflash_config` of a DFlash 2 draft checkpoint."""
    return SimpleNamespace(
        block_size=block_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
        resolve_block_size=lambda default=None: (
            block_size if block_size is not None else default
        ),
    )


def _make_args(**overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; the hook is invoked
    # directly, same as the other unit/spec ServerArgs tests.
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = "DFLASH"
    args.speculative_draft_model_path = "dummy-draft"
    args.device = "cuda"
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen3NextForCausalLM"],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _resolve(args: ServerArgs, draft_config=None, *, readable=True) -> ServerArgs:
    """Run the speculative hook with every draft-checkpoint read stubbed out.

    Two independent reads have to go: `_resolve_speculative_algorithm_alias` calls
    `get_config` on the draft path unconditionally (to sniff Gemma4 assistant drafts), and
    DFLASH's own width resolution parses `dflash_config`. Left real, both would hit the HF
    hub for a nonexistent repo.
    """

    def fake_load(server_args, *, required):
        return draft_config if draft_config is not None else _draft_config()

    hf_get_config = mock.patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=SimpleNamespace(architectures=["DFlash2DraftModel"]),
    )
    if readable:
        dflash_read = mock.patch(f"{_HOOK}._load_dflash_draft_config", fake_load)
    else:
        # Break the parse one level down, inside the real `_load_dflash_draft_config`, so its
        # own required/optional branching is what the test exercises. A malformed
        # `dflash_config` is the realistic failure -- an unreadable draft path would already
        # have failed in the algorithm-alias sniff before DFLASH's hook runs.
        dflash_read = mock.patch(
            "sglang.srt.speculative.dflash_utils.parse_dflash_draft_config",
            side_effect=ValueError("malformed dflash_config"),
        )
    with hf_get_config, dflash_read:
        handle_speculative_decoding(args)
    return args


def _tree_args(*, tree_width=2, **overrides) -> ServerArgs:
    """A tree-width configuration that passes every admission check, so a test can flip
    exactly one field and attribute the resulting error to that field."""
    defaults = dict(
        speculative_dflash_tree_width=tree_width,
        speculative_dflash_block_size=8,
        attention_backend="triton",
        page_size=1,
        disable_decode_cuda_graph=True,
    )
    defaults.update(overrides)
    return _make_args(**defaults)


class TestDflashChainWidthUnchanged(CustomTestCase):
    """Tree width 1 (the default) must resolve exactly as DFLASH did before the split."""

    def test_num_draft_tokens_alias_still_sets_block_size(self):
        # The pre-split launch path: --speculative-num-draft-tokens 8 means block width 8.
        args = _resolve(_make_args(speculative_num_draft_tokens=8))

        self.assertEqual(args.speculative_dflash_block_size, 8)
        self.assertEqual(args.speculative_num_draft_tokens, 8)
        self.assertEqual(args.speculative_eagle_topk, 1)

    def test_explicit_block_size_alone_resolves(self):
        args = _resolve(_make_args(speculative_dflash_block_size=8))

        self.assertEqual(args.speculative_dflash_block_size, 8)
        self.assertEqual(args.speculative_num_draft_tokens, 8)
        self.assertEqual(args.speculative_eagle_topk, 1)

    def test_block_size_inferred_from_draft_checkpoint(self):
        args = _resolve(_make_args(), _draft_config(block_size=5))

        self.assertEqual(args.speculative_dflash_block_size, 5)
        self.assertEqual(args.speculative_num_draft_tokens, 5)

    def test_matching_alias_and_explicit_block_size_agree(self):
        args = _resolve(
            _make_args(speculative_num_draft_tokens=8, speculative_dflash_block_size=8)
        )

        self.assertEqual(args.speculative_dflash_block_size, 8)
        self.assertEqual(args.speculative_num_draft_tokens, 8)

    def test_conflicting_alias_and_explicit_block_size_raise(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            _resolve(
                _make_args(
                    speculative_num_draft_tokens=4, speculative_dflash_block_size=8
                )
            )

    def test_unreadable_draft_config_falls_back_to_default_block_size(self):
        # Chain width tolerates an unreadable checkpoint (it only needs a block width);
        # tree width does not, which is asserted separately below.
        args = _resolve(_make_args(), readable=False)

        self.assertEqual(args.speculative_dflash_block_size, 16)
        self.assertEqual(args.speculative_num_draft_tokens, 16)


class TestDflashTreeWidthDerivation(CustomTestCase):
    def test_verify_width_grows_with_tree_width(self):
        for tree_width, verify_width in ((1, 8), (2, 15), (4, 29), (8, 57)):
            with self.subTest(tree_width=tree_width):
                args = _resolve(_tree_args(tree_width=tree_width))

                self.assertEqual(args.speculative_dflash_block_size, 8)
                self.assertEqual(args.speculative_num_draft_tokens, verify_width)
                self.assertEqual(args.speculative_eagle_topk, tree_width)

    def test_non_power_of_two_block_size(self):
        # gamma = 4 -> 1 + 4 * 2. Guards against a derivation that assumes a power of two.
        args = _resolve(_tree_args(tree_width=2, speculative_dflash_block_size=5))

        self.assertEqual(args.speculative_num_draft_tokens, 9)

    def test_num_steps_stays_one(self):
        # Widening the verify window must not leak into num_steps: accept_index is sized from
        # max_tree_depth, and generic accounting still assumes DFLASH takes one step.
        args = _resolve(_tree_args(tree_width=4))

        self.assertEqual(args.speculative_num_steps, 1)


class TestDflashTreeWidthRejections(CustomTestCase):
    def test_eagle_topk_is_rejected(self):
        # Even the value DFLASH used to force (1): the flag has no DFLASH meaning at all, so
        # accepting it would leave two ways to spell the width.
        for topk in (1, 4):
            with self.subTest(topk=topk):
                with self.assertRaisesRegex(ValueError, "dflash-tree-width"):
                    _resolve(_make_args(speculative_eagle_topk=topk))

    def test_num_draft_tokens_rejected_with_a_wide_tree(self):
        with self.assertRaisesRegex(ValueError, "speculative-dflash-block-size"):
            _resolve(_tree_args(tree_width=2, speculative_num_draft_tokens=15))

    def test_zero_tree_width_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            _resolve(_tree_args(tree_width=0))

    def test_tree_width_above_selector_top_k_rejected(self):
        with self.assertRaisesRegex(ValueError, "selector_top_k"):
            _resolve(_tree_args(tree_width=8), _draft_config(selector_top_k=4))

    def test_checkpoint_without_selector_rejected(self):
        with self.assertRaisesRegex(ValueError, "candidate selector"):
            _resolve(
                _tree_args(tree_width=2),
                _draft_config(selector_rank=0, selector_top_k=0),
            )

    def test_unreadable_draft_config_rejected_with_a_wide_tree(self):
        # The selector checks live only in the hook, so a swallowed read would turn both of
        # them into silent passes.
        with self.assertRaisesRegex(ValueError, "selector_top_k"):
            _resolve(_tree_args(tree_width=2), readable=False)


class TestDflashTreeAdmissionChecks(CustomTestCase):
    """The W>1 gates. DFLASH cannot borrow EAGLE's `_PAGE_TREE_SPEC_BACKENDS` check (that one
    lives in the EAGLE branch of the hook), so every one of these is load-bearing."""

    def test_mask_capable_backends_are_accepted(self):
        # The whitelist direction guard: triton is the SM100 default for hybrid-GDN models, so
        # a whitelist written as the complement of the skip-custom-mask set would reject the
        # one backend this feature actually runs on.
        for backend in ("triton", "flashinfer", "fa3"):
            with self.subTest(backend=backend):
                args = _resolve(_tree_args(attention_backend=backend))

                self.assertEqual(args.speculative_eagle_topk, 2)

    def test_trtllm_mha_rejected_as_silently_wrong(self):
        with self.assertRaisesRegex(ValueError, "silently"):
            _resolve(_tree_args(attention_backend="trtllm_mha"))

    def test_mask_incapable_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "custom tree mask"):
            _resolve(_tree_args(attention_backend="aiter"))

    def test_prefill_and_decode_backend_slots_are_checked(self):
        # A tree verify lands on whichever slot the forward resolves to, so an incompatible
        # override in either slot has to be caught even when --attention-backend is fine.
        for slot in ("prefill_attention_backend", "decode_attention_backend"):
            with self.subTest(slot=slot):
                with self.assertRaisesRegex(ValueError, "silently"):
                    _resolve(_tree_args(**{slot: "trtllm_mha"}))

    def test_paged_kv_rejected(self):
        with self.assertRaisesRegex(ValueError, "page-size"):
            _resolve(_tree_args(page_size=16))

    def test_replayssm_spec_rejected(self):
        # Its own topk guard runs in _handle_linear_attn_backend, before the speculative hook
        # assigns topk, so it sees None and lets this through.
        with self.assertRaisesRegex(ValueError, "replayssm-spec"):
            _resolve(_tree_args(enable_linear_replayssm_spec=True))

    def test_decode_cuda_graph_must_be_disabled(self):
        with self.assertRaisesRegex(ValueError, "disable-decode-cuda-graph"):
            _resolve(_tree_args(disable_decode_cuda_graph=False))

    def test_disable_cuda_graph_also_satisfies_the_graph_gate(self):
        args = _resolve(
            _tree_args(disable_decode_cuda_graph=False, disable_cuda_graph=True)
        )

        self.assertEqual(args.speculative_eagle_topk, 2)

    def test_simulated_real_draft_tokens_rejected(self):
        with envs.SGLANG_SIMULATE_ACC_LEN.override(
            3.0
        ), envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.override("real-draft-token"):
            with self.assertRaisesRegex(ValueError, "real-draft-token"):
                _resolve(_tree_args())

    def test_simulated_fixed_tokens_allowed(self):
        with envs.SGLANG_SIMULATE_ACC_LEN.override(
            3.0
        ), envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.override("fixed"):
            args = _resolve(_tree_args())

        self.assertEqual(args.speculative_eagle_topk, 2)

    def test_chain_width_is_exempt_from_every_gate(self):
        # The gates must not regress the single-path launch: today's DFLASH runs with decode
        # cuda graph on, and may run on a backend that cannot express a tree.
        args = _resolve(
            _make_args(
                speculative_num_draft_tokens=8,
                attention_backend="trtllm_mha",
                page_size=16,
                disable_decode_cuda_graph=False,
            )
        )

        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 8)


if __name__ == "__main__":
    unittest.main()
