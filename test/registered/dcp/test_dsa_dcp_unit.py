"""CPU unit tests for decode context parallel (DCP) on DSA models (Hopper).

Covers the parts of the DSA + DCP composition that need no GPU:

* ``dcp_localize_topk_slots`` turns the virtual top-k slot table (identical on
  every rank, because the index-K cache is replicated) into the rows each rank
  physically owns: virtual slot ``v`` lives on rank ``v % dcp_size`` at row
  ``v // dcp_size``.
* ``_dsa_dcp_validation`` rejects unsupported DSA + DCP compositions at config
  time instead of letting them boot and read the wrong KV rows.
* Which forward modes return a partial output plus LSE for the DCP merge.

Usage:
    python -m pytest test_dsa_dcp_unit.py -v
    python test_dsa_dcp_unit.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import _dsa_dcp_validation
from sglang.srt.layers.attention.dsa.utils import dcp_localize_topk_slots
from sglang.srt.layers.attention.dsa_backend import (
    _should_return_dsa_dcp_lse_flashmla_kv,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HOPPER = dict(is_cuda=True, is_blackwell=False)
OVERRIDES_LOGGER = "sglang.srt.arg_groups.overrides"


class TestDcpLocalizeTopkSlots(CustomTestCase):
    def _virtual_slots(self, n_valid=2000, n_pad=48, pool=50_000, seed=0):
        g = torch.Generator().manual_seed(seed)
        valid = torch.randperm(pool, generator=g)[:n_valid].to(torch.int32)
        pad = torch.full((n_pad,), -1, dtype=torch.int32)
        return torch.cat([valid, pad]).view(1, -1)

    def test_every_valid_slot_maps_to_exactly_one_rank_row(self):
        slots = self._virtual_slots()
        valid = slots[slots >= 0].long()
        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                owners = torch.zeros_like(slots, dtype=torch.int64)
                for rank in range(dcp_size):
                    local = dcp_localize_topk_slots(slots, rank, dcp_size)
                    self.assertEqual(local.dtype, slots.dtype)
                    self.assertEqual(local.shape, slots.shape)
                    kept = local >= 0
                    owned = (slots >= 0) & (slots % dcp_size == rank)
                    self.assertTrue(torch.equal(kept, owned))
                    self.assertTrue(torch.equal(local[kept], slots[kept] // dcp_size))
                    owners += kept
                # Each valid column is attended by exactly one rank, so the
                # LSE merge of the partial outputs covers the top-k set once.
                self.assertTrue(torch.equal(owners[slots >= 0], torch.ones_like(valid)))
                self.assertTrue(torch.equal(owners[slots < 0], torch.zeros(48).long()))

    def test_input_table_is_not_modified(self):
        # The virtual table is reused by the following index_topk_freq - 1
        # layers, so localizing must not write into it.
        slots = self._virtual_slots()
        before = slots.clone()
        dcp_localize_topk_slots(slots, 3, 8)
        self.assertTrue(torch.equal(slots, before))

    def test_identity_without_dcp(self):
        slots = self._virtual_slots()
        self.assertIs(dcp_localize_topk_slots(slots, 0, 1), slots)


def _view(num_attention_heads=128, index_kpool=1, **overrides):
    """A resolved server-args view for DeepSeek-V3.2 with the supported
    DSA + DCP composition; ``overrides`` replaces individual fields."""
    hf_config = SimpleNamespace(
        architectures=["DeepseekV32ForCausalLM"],
        index_topk=2048,
        num_attention_heads=num_attention_heads,
        index_kpool=index_kpool,
    )
    fields = dict(
        dcp_size=8,
        dcp_comm_backend="a2a",
        dcp_replicate_q_proj=False,
        kv_cache_dtype="fp8_e4m3",
        dsa_prefill_backend="flashmla_kv",
        dsa_decode_backend="flashmla_kv",
        page_size=64,
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
        hicache_write_policy="write_through",
        hicache_size=0,
        enable_lmcache=False,
        enable_hisparse=False,
        enable_prefill_cp=False,
        enable_dsa_prefill_context_parallel=False,
        enable_mixed_chunk=False,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        disaggregation_mode="null",
    )
    fields.update(overrides)
    return SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(hf_config=hf_config), **fields
    )


class TestDsaDcpValidation(CustomTestCase):
    def assert_accepted(self, view):
        # The info log is emitted only after every check ran, so a pass that
        # returned early (not a DSA model, not Hopper) cannot satisfy this.
        with override_platform(**HOPPER):
            with self.assertLogs(OVERRIDES_LOGGER, "INFO") as logs:
                self.assertEqual(_dsa_dcp_validation(view), {})
        self.assertIn("DCP enabled for DSA model", "\n".join(logs.output))

    def assert_rejected(self, view, *expected):
        with override_platform(**HOPPER):
            with self.assertRaises(ValueError) as ctx:
                _dsa_dcp_validation(view)
        for text in expected:
            self.assertIn(text, str(ctx.exception))

    def test_supported_composition_passes(self):
        self.assert_accepted(_view())

    def test_ignored_without_dcp_off_hopper_or_for_other_models(self):
        def bad_view(**overrides):
            return _view(kv_cache_dtype="auto", page_size=1, **overrides)

        dense = bad_view()
        dense.get_model_config = lambda: SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"], num_attention_heads=32
            )
        )
        cases = [
            (HOPPER, bad_view(dcp_size=1)),
            (dict(is_cuda=True, is_blackwell=True), bad_view()),
            (dict(is_cuda=False, is_blackwell=False), bad_view()),
            (HOPPER, dense),
        ]
        for platform, view in cases:
            with self.subTest(platform=platform, dcp_size=view.dcp_size):
                with override_platform(**platform):
                    self.assertEqual(_dsa_dcp_validation(view), {})

    def test_rejects_each_unsupported_setting(self):
        cases = [
            (_view(kv_cache_dtype="auto"), "kv_cache_dtype='auto'"),
            (_view(dsa_prefill_backend="flashmla_sparse"), "dsa_prefill_backend"),
            (_view(dsa_decode_backend="fa3"), "dsa_decode_backend"),
            (_view(index_kpool=4), "index_kpool"),
            (_view(num_attention_heads=60), "num_attention_heads=60"),
            (_view(enable_lmcache=True), "--enable-lmcache"),
            (_view(enable_hisparse=True), "--enable-hisparse"),
            (_view(enable_prefill_cp=True), "--enable-prefill-cp"),
            (
                _view(enable_dsa_prefill_context_parallel=True),
                "DSA prefill context parallel",
            ),
            (_view(enable_mixed_chunk=True), "--enable-mixed-chunk"),
            (_view(speculative_algorithm="NGRAM"), "speculative_algorithm='NGRAM'"),
            (
                _view(speculative_algorithm="EAGLE", speculative_eagle_topk=4),
                "speculative",
            ),
            (
                _view(enable_hierarchical_cache=True, hicache_storage_backend="file"),
                "HiCache",
            ),
            (_view(disaggregation_mode="decode"), "PD disaggregation"),
            (_view(dcp_replicate_q_proj=True), "--dcp-replicate-q-proj"),
            (_view(page_size=32), "page_size=32"),
        ]
        for view, expected in cases:
            with self.subTest(expected=expected):
                self.assert_rejected(view, expected)

    def test_reports_every_problem_at_once(self):
        self.assert_rejected(
            _view(kv_cache_dtype="auto", enable_mixed_chunk=True, page_size=32),
            "kv_cache_dtype='auto'",
            "--enable-mixed-chunk",
            "page_size=32",
        )


class TestDsaDcpLseMergePhases(CustomTestCase):
    MERGED = (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY)
    NOT_MERGED = (ForwardMode.MIXED, ForwardMode.IDLE, ForwardMode.SPLIT_PREFILL)

    def test_flashmla_kv_returns_lse_only_in_merged_phases(self):
        for mode in self.MERGED + self.NOT_MERGED:
            with self.subTest(mode=mode):
                self.assertEqual(
                    _should_return_dsa_dcp_lse_flashmla_kv(
                        forward_mode=mode, dcp_enabled=True
                    ),
                    mode in self.MERGED,
                )
                self.assertFalse(
                    _should_return_dsa_dcp_lse_flashmla_kv(
                        forward_mode=mode, dcp_enabled=False
                    )
                )

    def test_plain_extend_merges_only_for_dsa(self):
        # Dense MLA extend gathers the prefix KV instead of merging partials.
        def batch(mode):
            return SimpleNamespace(forward_mode=mode)

        with patch.object(
            forward_mla, "get_parallel", return_value=SimpleNamespace(dcp_enabled=True)
        ):
            for mode in self.MERGED + self.NOT_MERGED:
                with self.subTest(mode=mode):
                    self.assertEqual(
                        forward_mla.is_dcp_lse_merge_phase(batch(mode), use_dsa=True),
                        mode in self.MERGED,
                    )
            self.assertFalse(
                forward_mla.is_dcp_lse_merge_phase(
                    batch(ForwardMode.EXTEND), use_dsa=False
                )
            )
            self.assertTrue(
                forward_mla.is_dcp_lse_merge_phase(
                    batch(ForwardMode.DECODE), use_dsa=False
                )
            )
        with patch.object(
            forward_mla,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=False),
        ):
            self.assertFalse(
                forward_mla.is_dcp_lse_merge_phase(
                    batch(ForwardMode.EXTEND), use_dsa=True
                )
            )


if __name__ == "__main__":
    unittest.main()
