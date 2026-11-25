"""CPU tests for the opt-in Kimi-K3.1 sparse-MLA prefill reader.

The FlashMLA operators are replaced by torch reference implementations with the same
calling convention (dequant of V32 pages, deterministic union pack with ownership bits
26..30, sparse attention with ``heads_per_group``), so the module's own work is checked
on CPU: argument screening, slicing and views, the T mod G tail launch, -1 padded and
over-long top-k tables, workspace reuse, the overlap guard, fallback when the installed
sgl-flash-mla lacks the operators or one of them fails, flag parsing, constructor
eligibility and dispatch precedence. The real kernels need an SM100-family GPU and an
sgl-flash-mla that carries them.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.kernels.ops.attention.dsa import k31_prefill_reader as reader
from sglang.srt.arg_groups.fields.exec_ import ExecKernel
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _k31_sparse_prefill_reader_decision,
    _resolve_k31_sparse_prefill_reader,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SHIFT = 26
SLOT_MASK = (1 << SHIFT) - 1


def ref_dequant(kv_cache, out=None, rows=None):
    """V32 page rows -> [rows, 1, 576] bf16. The stand-in keeps the payload in the first
    576 bytes of a row so a test can state the expected values directly."""
    assert rows is None
    flat = kv_cache.reshape(-1, 656)[:, :576].to(torch.float32)
    if out is None:
        out = torch.empty(flat.shape[0], 1, 576, dtype=torch.bfloat16)
    out.copy_((flat / 64.0 - 2.0).to(out.dtype).view(-1, 1, 576))
    return out


def ref_pack(indices, group, packed, lengths):
    """Sorted union of each group's valid ids; bit 26 + g marks token g as an owner."""
    n = indices.shape[0] // group
    assert (
        indices.shape[0] == n * group and packed.shape[0] >= n and lengths.shape[0] >= n
    )
    for row in range(n):
        owners = {}
        for g in range(group):
            for slot in indices[row * group + g].tolist():
                if slot >= 0:
                    owners[slot] = owners.get(slot, 0) | (1 << (SHIFT + g))
        merged = [slot | bits for slot, bits in sorted(owners.items())]
        assert len(merged) <= packed.shape[1]
        packed[row].fill_(-1)
        packed[row, : len(merged)] = torch.tensor(merged, dtype=torch.int32)
        lengths[row] = len(merged)
    return packed, lengths


def ref_attention(
    q, kv, indices, sm_scale, heads_per_group=0, topk_length=None, out=None
):
    """q [N, R, 576], kv [rows, 1, 576], indices [N, 1, W], topk_length [N].
    heads_per_group == 0: every row of q attends to the first topk_length valid ids.
    heads_per_group == h: q row r belongs to token r // h and attends to the ids that
    carry its ownership bit."""
    n, rows_q, _ = q.shape
    if out is None:
        out = torch.empty(n, rows_q, 512, dtype=q.dtype)
    keys = kv[:, 0].double()
    step = heads_per_group or rows_q
    for i in range(n):
        ids = indices[i, 0, : int(topk_length[i])]
        ids = ids[ids >= 0]
        for first in range(0, rows_q, step):
            mine = ids
            if heads_per_group:
                mine = ids[(ids & (1 << (SHIFT + first // step))) != 0] & SLOT_MASK
            out[i, first : first + step] = _attend(
                keys[mine.long()], q[i, first : first + step], sm_scale
            )
    return out


def _attend(k, q_rows, sm_scale):
    """[rows, 512] bf16: softmax(q k^T * scale) k[:, :512] in float64; zeros without keys."""
    if k.shape[0] == 0:
        return torch.zeros(q_rows.shape[0], 512, dtype=torch.bfloat16)
    w = torch.softmax(q_rows.double() @ k.T * sm_scale, dim=1)
    return (w @ k[:, :512]).to(torch.bfloat16)


def direct_reference(q_all, kv_cache, indices, sm_scale):
    """Per token, straight from the definition."""
    kv = ref_dequant(kv_cache)
    keys = kv[:, 0].double()
    T, h, _ = q_all.shape
    out = torch.zeros(T, 1, h, 512, dtype=torch.bfloat16)
    for t in range(T):
        mine = indices[t][indices[t] >= 0].long()
        out[t, 0] = _attend(keys[mine], q_all[t], sm_scale)
    return out


def make_inputs(
    T=13,
    h=12,
    topk=128,
    pages=4,
    seed=0,
    extra_rows=0,
    valid=None,
    disjoint=False,
    split_q=True,
):
    gen = torch.Generator().manual_seed(seed)
    rows = pages * 64
    q = (torch.randn(T, h, 576, generator=gen) * 0.2).to(torch.bfloat16)
    kv_cache = torch.randint(
        0, 256, (pages, 64, 1, 656), generator=gen, dtype=torch.uint8
    )
    indices = torch.full((T + extra_rows, topk), -1, dtype=torch.int32)
    for t in range(T):
        count = topk if valid is None else valid[t % len(valid)]
        if disjoint:
            start = (t * topk) % rows
            ids = (torch.arange(count) + start) % rows
        else:
            ids = torch.randperm(rows, generator=gen)[:count]
        indices[t, :count] = ids.to(torch.int32)
    if split_q:
        return (
            q[..., :512].contiguous(),
            q[..., 512:].contiguous(),
            kv_cache,
            indices,
            q,
        )
    return q, None, kv_cache, indices, q


class _ReaderCase(CustomTestCase):
    def setUp(self):
        # The references are many small float64 products; one thread keeps them fast on
        # hosts with far more cores than the test is allowed to use.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        reader._reset_for_tests()
        self.addCleanup(reader._reset_for_tests)
        self.ops = (
            MagicMock(side_effect=ref_attention),
            MagicMock(side_effect=ref_pack),
            MagicMock(side_effect=ref_dequant),
        )
        p = patch.object(reader, "_load", return_value=self.ops)
        p.start()
        self.addCleanup(p.stop)

    def run_reader(self, pack=True, **kw):
        q, q_rope, kv_cache, indices, q_all = make_inputs(**kw)
        out = reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05, pack=pack)
        return out, direct_reference(q_all, kv_cache, indices, 0.05)

    def assert_close(self, out, ref):
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2),
            (out.float() - ref.float()).abs().max(),
        )


class TestNumerics(_ReaderCase):
    def test_unpacked_matches_the_definition(self):
        out, ref = self.run_reader(pack=False)
        self.assert_close(out, ref)
        self.assertEqual(self.ops[0].call_count, 1)
        self.assertEqual(self.ops[0].call_args.args[4], 0)  # heads_per_group
        self.ops[1].assert_not_called()

    def test_packed_with_tail_matches_the_definition(self):
        out, ref = self.run_reader(
            pack=True, T=13
        )  # G = 5: two groups, three tail rows
        self.assert_close(out, ref)
        packed_call, tail_call = self.ops[0].call_args_list
        self.assertEqual(tuple(packed_call.args[0].shape), (2, 60, 576))
        self.assertEqual(packed_call.args[4], 12)
        self.assertEqual(tuple(tail_call.args[0].shape), (3, 12, 576))
        self.assertEqual(tail_call.args[4], 0)

    def test_packed_without_tail_and_single_tensor_q(self):
        out, ref = self.run_reader(pack=True, T=10, split_q=False)
        self.assert_close(out, ref)
        self.assertEqual(self.ops[0].call_count, 1)

    def test_fewer_tokens_than_a_group_runs_the_tail_only(self):
        out, ref = self.run_reader(pack=True, T=3)
        self.assert_close(out, ref)
        self.ops[1].assert_not_called()

    def test_minus_one_padded_rows(self):
        out, ref = self.run_reader(pack=True, T=11, valid=[128, 64, 1, 0, 37])
        self.assert_close(out, ref)
        out, ref = self.run_reader(pack=False, T=11, valid=[128, 64, 1, 0, 37])
        self.assert_close(out, ref)

    def test_top_k_table_longer_than_the_batch(self):
        # DP padding hands a table with more rows than query tokens; the extra rows are ignored.
        for pack in (True, False):
            out, ref = self.run_reader(pack=pack, T=13, extra_rows=7)
            self.assert_close(out, ref[:13])
            for call in self.ops[0].call_args_list:
                self.assertEqual(call.args[0].shape[0], call.args[2].shape[0])

    def test_head_counts(self):
        for h, group in ((1, 5), (8, 5), (16, 4), (32, 2)):
            self.ops[0].reset_mock()
            out, ref = self.run_reader(pack=True, T=2 * group + 1, h=h)
            self.assert_close(out, ref)
            self.assertEqual(self.ops[0].call_args_list[0].args[0].shape[1], group * h)

    def test_packed_equals_unpacked(self):
        packed, _ = self.run_reader(pack=True, T=15, seed=3)
        unpacked, _ = self.run_reader(pack=False, T=15, seed=3)
        self.assert_close(packed, unpacked)

    def test_workspace_is_reused_and_grows(self):
        self.run_reader(pack=True, T=10)
        first = reader._WS[(None, "pi")]
        self.run_reader(pack=True, T=5)
        self.assertIs(reader._WS[(None, "pi")], first)
        self.run_reader(pack=True, T=20)
        self.assertGreaterEqual(reader._WS[(None, "pi")].shape[0], 4)
        self.assertIsNot(reader._WS[(None, "pi")], first)

    def test_capture_gets_private_workspace(self):
        like = torch.empty(1)
        shared = reader._ws("pi", like, (4, 64), torch.int32)
        private = reader._ws("pi", like, (4, 64), torch.int32, capturing=True)
        self.assertIsNot(private, shared)
        self.assertIs(reader._WS[(None, "pi")], shared)


class TestScreening(_ReaderCase):
    def _declines(self, why, **changes):
        q, q_rope, kv_cache, indices, _ = make_inputs()
        args = dict(q=q, q_rope=q_rope, kv_cache=kv_cache, indices=indices)
        args.update(changes)
        out = reader.k31_prefill_reader(
            args["q"], args["q_rope"], args["kv_cache"], args["indices"], 0.05
        )
        self.assertIsNone(out, why)
        self.assertIn(why, reader.reader_stats()["declined_reasons"])
        for op in self.ops:
            op.assert_not_called()

    def test_declined_calls(self):
        q, q_rope, kv_cache, indices, q_all = make_inputs()
        self._declines("q is not a 3-d bf16 tensor", q=q.float())
        self._declines("empty batch", q=q[:0], q_rope=q_rope[:0])
        self._declines(
            "head count outside 1..32",
            q=q.repeat(1, 4, 1),
            q_rope=q_rope.repeat(1, 4, 1),
        )
        self._declines("query width is not 512 + 64", q=q_all)
        self._declines("q_rope does not match q", q_rope=q_rope[:, :6])
        self._declines("KV cache is not FP8", kv_cache=kv_cache.to(torch.bfloat16))
        self._declines(
            "KV buffer is not contiguous V32 pages", kv_cache=kv_cache[:, :32]
        )
        self._declines("KV buffer is not contiguous V32 pages", kv_cache=kv_cache[::2])
        self._declines(
            "top-k ids are not contiguous int32 rows", indices=indices.long()
        )
        self._declines(
            "top-k ids are not contiguous int32 rows", indices=indices[:, ::2]
        )
        self._declines("fewer top-k rows than query tokens", indices=indices[:5])
        self._declines(
            "top-k width is not a multiple of 64 in 128..2048",
            indices=indices[:, :64].contiguous(),
        )
        self._declines(
            "top-k width is not a multiple of 64 in 128..2048",
            indices=indices[:, :100].contiguous(),
        )
        self.assertEqual(reader.reader_stats()["declined"], 13)

    def test_size_limit(self):
        q, q_rope, kv_cache, indices, _ = make_inputs()
        out = reader.k31_prefill_reader(
            q, q_rope, kv_cache, indices, 0.05, max_bf16_bytes=4 * 64 * 1152 - 1
        )
        self.assertIsNone(out)
        out = reader.k31_prefill_reader(
            q, q_rope, kv_cache, indices, 0.05, max_bf16_bytes=4 * 64 * 1152
        )
        self.assertIsNotNone(out)

    def test_wide_top_k_runs_unpacked(self):
        # 5 tokens x 2496 ids would exceed the packed row; the module keeps such calls unpacked.
        q, q_rope, kv_cache, indices, _ = make_inputs(T=5, h=12, topk=2048, pages=40)
        self.assertIsNotNone(
            reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05, pack=True)
        )
        self.assertEqual(reader.reader_stats()["packed"], 1)
        with patch.object(reader, "_MAX_GROUP", 7):
            q, q_rope, kv_cache, indices, _ = make_inputs(T=7, h=8, topk=2048, pages=40)
            reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05, pack=True)
        self.assertEqual(reader.reader_stats()["unpacked"], 1)


class TestFallback(CustomTestCase):
    def setUp(self):
        reader._reset_for_tests()
        self.addCleanup(reader._reset_for_tests)

    def test_wheel_without_the_operators_keeps_stock_with_one_warning(self):
        # sgl-flash-mla before the token-packed prefill: the capability check says no
        old_wheel = SimpleNamespace(supports_sparse_prefill_token_packing=lambda: False)
        with patch.object(reader, "_flash_mla", return_value=old_wheel):
            with self.assertLogs(reader.logger, level="WARNING") as logs:
                self.assertFalse(reader.k31_prefill_reader_available())
                self.assertFalse(reader.k31_prefill_reader_available())
        self.assertEqual(len(logs.output), 1)
        self.assertIn("unavailable", logs.output[0])
        self.assertIn("flash_mla_sparse_fwd_packed", logs.output[0])

    def test_missing_wheel_keeps_stock(self):
        with patch.object(
            reader, "_flash_mla", side_effect=ImportError("No module named 'flash_mla'")
        ):
            with self.assertLogs(reader.logger, level="WARNING"):
                self.assertFalse(reader.k31_prefill_reader_available())

    def test_wheel_operators_are_found(self):
        wheel = SimpleNamespace(
            supports_sparse_prefill_token_packing=lambda: True,
            flash_mla_sparse_fwd_packed=1,
            flash_mla_sparse_pack_indices=2,
            flash_mla_dequant_v32_kv=3,
        )
        with patch.object(reader, "_flash_mla", return_value=wheel):
            self.assertTrue(reader.k31_prefill_reader_available())
            self.assertEqual(reader._load(), (1, 2, 3))

    def test_reference_operators_follow_the_wheel_signatures(self):
        # The stand-ins of this file are called the way the reader calls the wheel:
        # flash_mla_sparse_fwd_packed(q, kv, indices, sm_scale, heads_per_group, topk_length, out),
        # flash_mla_sparse_pack_indices(indices, group, packed, lengths), flash_mla_dequant_v32_kv(kv, out).
        ops = (
            MagicMock(side_effect=ref_attention),
            MagicMock(side_effect=ref_pack),
            MagicMock(side_effect=ref_dequant),
        )
        q, q_rope, kv_cache, indices, _ = make_inputs(T=13)
        with patch.object(reader, "_load", return_value=ops):
            self.assertIsNotNone(
                reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05)
            )
        packed_call, tail_call = ops[0].call_args_list
        self.assertEqual(len(packed_call.args), 7)
        self.assertEqual(packed_call.args[3], 0.05)
        self.assertEqual(
            tuple(packed_call.args[5].shape), (2,)
        )  # lengths of the packed lists
        self.assertEqual(tuple(packed_call.args[6].shape), (2, 60, 512))
        self.assertEqual(tuple(tail_call.args[5].shape), (3,))
        self.assertEqual(len(ops[1].call_args.args), 4)
        self.assertEqual(len(ops[2].call_args.args), 2)

    def test_runtime_error_disables_the_reader(self):
        failing = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        q, q_rope, kv_cache, indices, _ = make_inputs()
        with patch.object(
            reader, "_load", return_value=(failing, ref_pack, ref_dequant)
        ):
            with self.assertLogs(reader.logger, level="ERROR"):
                self.assertIsNone(
                    reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05)
                )
            self.assertIsNone(
                reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05)
            )
        self.assertEqual(
            failing.call_count, 1
        )  # the second call never reaches the operators
        self.assertIn("out of memory", reader.reader_stats()["disabled"])

    def test_env_defaults(self):
        self.assertEqual(envs.SGLANG_K31_PREFILL_READER_MAX_UNION_FRACTION.get(), 0.8)


class TestOverlapGuard(_ReaderCase):
    def _guard(self, **kw):
        return reader._OverlapGuard(0.8, **kw)

    def test_stays_packed_on_shared_keys(self):
        guard = self._guard(probe_every=1)
        for _ in range(5):
            self.assertTrue(guard.want_packed(False))
            guard.observe(torch.tensor([300, 310]), torch.tensor([128] * 10), False)
        self.assertAlmostEqual(guard.last_fraction, 610 / 1280)
        self.assertEqual(guard.switches, 0)

    def test_switches_to_unpacked_and_probes_again(self):
        guard = self._guard(probe_every=4, hold=3)
        self.assertTrue(guard.want_packed(False))
        guard.observe(torch.tensor([630]), torch.tensor([128] * 5), False)  # 0.98
        self.assertEqual([guard.want_packed(False) for _ in range(3)], [False] * 3)
        self.assertEqual(guard.switches, 1)
        self.assertTrue(guard.want_packed(False))  # hold over: packed again ...
        guard.observe(
            torch.tensor([320]), torch.tensor([128] * 5), False
        )  # ... and measured at once
        self.assertIsNotNone(guard.pending)
        self.assertTrue(guard.want_packed(False))
        self.assertAlmostEqual(guard.last_fraction, 0.5)

    def test_probes_only_every_nth_packed_call(self):
        guard = self._guard(probe_every=4)
        probes = 0
        for _ in range(12):
            guard.want_packed(False)
            guard.observe(torch.tensor([10]), torch.tensor([100]), False)
            probes += guard.pending is not None
        self.assertEqual(probes, 3)

    def test_capture_neither_reads_nor_records(self):
        guard = self._guard(probe_every=1, hold=2)
        guard.observe(torch.tensor([10]), torch.tensor([10]), True)
        self.assertIsNone(guard.pending)
        guard.observe(torch.tensor([10]), torch.tensor([10]), False)
        pending = guard.pending
        self.assertTrue(guard.want_packed(True))  # not read inside a capture
        self.assertIs(guard.pending, pending)
        self.assertFalse(guard.want_packed(False))
        self.assertFalse(guard.want_packed(True))  # a capture keeps the current choice
        self.assertEqual(guard.unpacked_left, 1)

    def test_disabled_guard_never_reads(self):
        guard = reader._OverlapGuard(1.0, probe_every=1)
        guard.observe(torch.tensor([10]), torch.tensor([10]), False)
        self.assertIsNone(guard.pending)
        self.assertTrue(guard.want_packed(False))

    def test_all_padding_group_does_not_divide_by_zero(self):
        guard = self._guard(probe_every=1)
        guard.observe(torch.tensor([0]), torch.tensor([0, 0]), False)
        self.assertTrue(guard.want_packed(False))
        self.assertEqual(guard.last_fraction, 0.0)

    def test_reader_leaves_packing_when_rows_share_no_keys(self):
        reader._GUARD = reader._OverlapGuard(0.8, probe_every=1, hold=2)
        kinds = []
        for _ in range(5):
            out, ref = self.run_reader(pack=True, T=10, pages=40, disjoint=True)
            self.assert_close(out, ref)
            stats = reader.reader_stats()
            kinds.append((stats["packed"], stats["unpacked"]))
        self.assertEqual(kinds, [(1, 0), (1, 1), (1, 2), (2, 2), (2, 3)])
        self.assertEqual(reader.reader_stats()["last_union_fraction"], 1.0)

    def test_reader_keeps_packing_on_overlapping_rows(self):
        reader._GUARD = reader._OverlapGuard(0.8, probe_every=1, hold=2)
        for _ in range(4):
            self.run_reader(
                pack=True, T=10, pages=3
            )  # 192 rows, 128 ids each: heavy overlap
        stats = reader.reader_stats()
        self.assertEqual((stats["packed"], stats["unpacked"]), (4, 0))
        self.assertLess(stats["last_union_fraction"], 0.5)


def _resolve(**changes):
    args = dict(
        requested="packed",
        deterministic=False,
        is_cuda=True,
        is_sm100=True,
        shards_index_k=True,
        prefill_cp=False,
        index_kpool=1,
        kv_cache_store_fp8=True,
        kv_format="V32",
        physical_page_size=64,
        local_num_q_heads=12,
        hisparse=False,
        available=lambda: True,
    )
    args.update(changes)
    mode, why = _k31_sparse_prefill_reader_decision(**args)
    assert mode == _resolve_k31_sparse_prefill_reader(**args)
    assert why
    return mode


def _why(**changes):
    args = dict(
        requested="packed",
        deterministic=False,
        is_cuda=True,
        is_sm100=True,
        shards_index_k=True,
        prefill_cp=False,
        index_kpool=1,
        kv_cache_store_fp8=True,
        kv_format="V32",
        physical_page_size=64,
        local_num_q_heads=12,
        hisparse=False,
        available=lambda: True,
        prefill_impl="flashmla_kv",
    )
    args.update(changes)
    return _k31_sparse_prefill_reader_decision(**args)


class TestEligibility(CustomTestCase):
    def test_flag_is_off_by_default(self):
        field = {f.name: f for f in msgspec.structs.fields(ExecKernel)}[
            "k31_sparse_prefill_reader"
        ]
        self.assertEqual(field.default, "off")

    def test_covered_deployment(self):
        self.assertEqual(_resolve(), "packed")
        self.assertEqual(_resolve(requested="unpacked"), "unpacked")
        self.assertEqual(_resolve(local_num_q_heads=32), "packed")

    def test_deterministic_inference_forces_unpacked(self):
        self.assertEqual(_resolve(deterministic=True), "unpacked")

    def test_off_never_asks_for_the_operators(self):
        available = MagicMock(side_effect=AssertionError)
        self.assertEqual(_resolve(requested="off", available=available), "off")

    def test_uncovered_deployments_stay_off_without_loading(self):
        available = MagicMock(return_value=True)
        for change in (
            dict(is_cuda=False),
            dict(is_sm100=False),
            dict(shards_index_k=False),
            dict(prefill_cp=True),
            dict(index_kpool=2),
            dict(kv_cache_store_fp8=False),
            dict(kv_format="V32_NO_ROPE"),
            dict(kv_format=None),
            dict(physical_page_size=32),
            dict(local_num_q_heads=48),
            dict(local_num_q_heads=0),
            dict(hisparse=True),
        ):
            with self.subTest(change=change):
                self.assertEqual(_resolve(available=available, **change), "off")
        available.assert_not_called()

    def test_missing_operators_stay_off(self):
        self.assertEqual(_resolve(available=lambda: False), "off")

    def test_other_dsa_backends_stay_off_without_looking_for_operators(self):
        # Kimi-K3.1 defaults to prims_ts on SM100-family GPUs: another KV layout,
        # and the extend never reaches the flashmla_kv branch the reader hooks.
        available = MagicMock(return_value=True)
        for backend in ("prims_ts", "trtllm", "flashmla_sparse", None):
            with self.subTest(backend=backend):
                mode, why = _why(prefill_impl=backend, available=available)
                self.assertEqual(mode, "off")
                self.assertIn("--dsa-prefill-backend", why)
                self.assertIn("flashmla_kv", why)
        available.assert_not_called()

    def test_every_off_decision_names_its_reason(self):
        cases = (
            (dict(is_cuda=False), "CUDA"),
            (dict(is_sm100=False), "SM100"),
            (dict(shards_index_k=False), "index-K"),
            (dict(prefill_cp=True), "prefill CP"),
            (dict(index_kpool=2), "K-pool is 2"),
            (dict(kv_cache_store_fp8=False), "fp8_ds_mla"),
            (dict(kv_format="V32_NO_ROPE"), "V32_NO_ROPE"),
            (dict(physical_page_size=32), "page size is 32"),
            (dict(local_num_q_heads=48), "48 heads"),
            (dict(hisparse=True), "HiSparse"),
            (dict(available=lambda: False), "sgl-flash-mla"),
        )
        for change, text in cases:
            with self.subTest(change=change):
                mode, why = _why(**change)
                self.assertEqual(mode, "off")
                self.assertIn(text, why)
        mode, why = _why(prefill_impl="prims_ts", kv_cache_store_fp8=False)
        self.assertIn("--dsa-prefill-backend", why)
        self.assertIn("fp8_ds_mla", why)  # every failed requirement is listed

    def test_active_decisions_say_so(self):
        self.assertEqual(_why(), ("packed", "all requirements met"))
        self.assertEqual(_why(requested="off")[0], "off")
        mode, why = _why(deterministic=True)
        self.assertEqual(mode, "unpacked")
        self.assertIn("deterministic", why)

    def test_constructor_logs_the_reason(self):
        import inspect

        source = inspect.getsource(DeepseekSparseAttnBackend.__init__)
        self.assertIn("requested %s, active %s (%s)", source)
        self.assertIn("_k31_sparse_prefill_reader_decision(", source)
        self.assertIn("prefill_impl=get_exec().kernel.dsa_prefill_backend", source)


class TestDispatch(CustomTestCase):
    def _backend(self, mode):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.k31_sparse_prefill_reader = mode
        backend.physical_page_size = 64
        backend.kv_cache_dim = 656
        return backend

    def _call(
        self,
        backend,
        forward_mode=ForwardMode.EXTEND,
        v_head_dim=512,
        q_rope=True,
        result="reader",
    ):
        layer = SimpleNamespace(v_head_dim=v_head_dim, scaling=0.05)
        batch = SimpleNamespace(forward_mode=forward_mode)
        kv = torch.zeros(2 * 64, 656, dtype=torch.uint8)
        with patch.object(reader, "k31_prefill_reader", return_value=result) as fn:
            out = backend._k31_prefill_reader_extend(
                layer,
                batch,
                "q_nope",
                "q_rope" if q_rope else None,
                "q_all",
                kv,
                "table",
            )
        return out, fn

    def test_off_never_calls_the_reader(self):
        out, fn = self._call(self._backend("off"))
        self.assertIsNone(out)
        fn.assert_not_called()

    def test_extend_goes_to_the_reader_with_the_gathered_buffer(self):
        out, fn = self._call(self._backend("packed"))
        self.assertEqual(out, "reader")
        q, q_rope, kv, table, scale = fn.call_args.args
        self.assertEqual((q, q_rope, table, scale), ("q_nope", "q_rope", "table", 0.05))
        self.assertEqual(tuple(kv.shape), (2, 64, 1, 656))
        self.assertTrue(fn.call_args.kwargs["pack"])
        out, fn = self._call(self._backend("unpacked"), q_rope=False)
        self.assertEqual(fn.call_args.args[0], "q_all")
        self.assertFalse(fn.call_args.kwargs["pack"])

    def test_decline_falls_through_to_stock(self):
        out, fn = self._call(self._backend("packed"), result=None)
        self.assertIsNone(out)
        fn.assert_called_once()

    def test_speculative_and_other_modes_keep_stock(self):
        for mode in ForwardMode:
            batch_mode = SimpleNamespace(forward_mode=mode)
            if batch_mode.forward_mode.is_extend_without_speculative():
                continue
            with self.subTest(mode=mode):
                out, fn = self._call(self._backend("packed"), forward_mode=mode)
                self.assertIsNone(out)
                fn.assert_not_called()
        out, fn = self._call(self._backend("packed"), v_head_dim=256)
        self.assertIsNone(out)
        fn.assert_not_called()

    def test_reader_is_tried_before_the_stock_token_packing(self):
        import inspect

        source = inspect.getsource(DeepseekSparseAttnBackend.forward_extend)
        branch = source[source.index('dsa_impl == "flashmla_kv"') :]
        self.assertLess(
            branch.index("_k31_prefill_reader_extend"),
            branch.index("self.flashmla_prefill_token_packing"),
        )
        self.assertLess(
            branch.index("if reader_out is not None"),
            branch.index("flashmla_kv_token_packed"),
        )


if __name__ == "__main__":
    unittest.main()
