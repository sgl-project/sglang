"""CPU tests for the opt-in Kimi-K3.1 sparse-MLA prefill reader.

The FlashMLA operators are replaced by torch stand-ins with the same calling convention;
the real kernels need an SM100-family GPU and an sgl-flash-mla that carries them.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.kernels.ops.attention.dsa import k31_prefill_reader as reader
from sglang.srt.arg_groups.fields.exec_ import ExecKernel
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
    """V32 page rows -> [rows, 1, 576] bf16; the payload is a row's first 576 bytes."""
    assert rows is None
    flat = kv_cache.reshape(-1, 656)[:, :576].to(torch.float32)
    if out is None:
        out = torch.empty(flat.shape[0], 1, 576, dtype=torch.bfloat16)
    out.copy_((flat / 64.0 - 2.0).to(out.dtype).view(-1, 1, 576))
    return out


def ref_pack(indices, group, packed, lengths):
    """Sorted union of each group's valid ids; bit 26 + g marks token g as an owner."""
    n = indices.shape[0] // group
    assert indices.shape[0] == n * group and min(len(packed), len(lengths)) >= n
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


def ref_attention(q, kv, indices, sm_scale, group_heads=0, topk_length=None, out=None):
    """q [N, R, 576], kv [rows, 1, 576], indices [N, 1, W], topk_length [N]. With
    group_heads h, q row r is token r // h and attends to the ids carrying its bit."""
    n, rows_q, _ = q.shape
    if out is None:
        out = torch.empty(n, rows_q, 512, dtype=q.dtype)
    keys = kv[:, 0].double()
    step = group_heads or rows_q
    for i in range(n):
        ids = indices[i, 0, : int(topk_length[i])]
        ids = ids[ids >= 0]
        for first in range(0, rows_q, step):
            mine = ids
            if group_heads:
                mine = ids[(ids & (1 << (SHIFT + first // step))) != 0] & SLOT_MASK
            rows = slice(first, first + step)
            out[i, rows] = _attend(keys[mine.long()], q[i, rows], sm_scale)
    return out


def _attend(k, q_rows, sm_scale):
    """[rows, 512] bf16: softmax(q k^T * scale) k[:, :512] in float64."""
    if k.shape[0] == 0:
        return torch.zeros(q_rows.shape[0], 512, dtype=torch.bfloat16)
    w = torch.softmax(q_rows.double() @ k.T * sm_scale, dim=1)
    return (w @ k[:, :512]).to(torch.bfloat16)


def direct_reference(q_all, kv_cache, indices, sm_scale):
    """Per token, straight from the definition."""
    keys = ref_dequant(kv_cache)[:, 0].double()
    T, h, _ = q_all.shape
    out = torch.zeros(T, 1, h, 512, dtype=torch.bfloat16)
    for t in range(T):
        mine = indices[t][indices[t] >= 0].long()
        out[t, 0] = _attend(keys[mine], q_all[t], sm_scale)
    return out


def make_inputs(T=13, h=12, topk=128, pages=4, pad_rows=0, valid=None, disjoint=False):
    """(q_nope, q_rope, kv_cache, top-k ids, q_all)."""
    gen = torch.Generator().manual_seed(0)
    rows = pages * 64
    q = (torch.randn(T, h, 576, generator=gen) * 0.2).to(torch.bfloat16)
    kv_cache = torch.randint(256, (pages, 64, 1, 656), generator=gen, dtype=torch.uint8)
    indices = torch.full((T + pad_rows, topk), -1, dtype=torch.int32)
    for t in range(T):
        count = topk if valid is None else valid[t % len(valid)]
        if disjoint:
            ids = (torch.arange(count) + (t * topk) % rows) % rows
        else:
            ids = torch.randperm(rows, generator=gen)[:count]
        indices[t, :count] = ids.to(torch.int32)
    return q[..., :512].contiguous(), q[..., 512:].contiguous(), kv_cache, indices, q


class TestReader(CustomTestCase):
    def setUp(self):
        # One thread keeps the many small float64 reference products fast on big hosts.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        reader._reset_for_tests()
        self.addCleanup(reader._reset_for_tests)
        fakes = (ref_attention, ref_pack, ref_dequant)
        self.ops = tuple(MagicMock(side_effect=fake) for fake in fakes)
        wheel = SimpleNamespace(
            supports_sparse_prefill_token_packing=lambda: True,
            flash_mla_sparse_fwd_packed=self.ops[0],
            flash_mla_sparse_pack_indices=self.ops[1],
            flash_mla_dequant_v32_kv=self.ops[2],
        )
        p = patch.object(reader, "_flash_mla", return_value=wheel)
        p.start()
        self.addCleanup(p.stop)

    def run_reader(self, pack=True, split_q=True, **kw):
        """One call, checked against the per-token reference."""
        q, q_rope, kv_cache, indices, q_all = make_inputs(**kw)
        if not split_q:
            q, q_rope = q_all, None
        out = reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05, pack=pack)
        ref = direct_reference(q_all, kv_cache, indices, 0.05)
        self.assertEqual((out.shape, out.dtype), (ref.shape, torch.bfloat16))
        self.assertTrue(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))

    def test_unpacked_is_one_launch(self):
        self.assertTrue(reader.k31_prefill_reader_available())
        self.run_reader(pack=False)
        self.assertEqual(self.ops[0].call_count, 1)
        self.assertEqual(self.ops[0].call_args.args[4], 0)  # heads_per_group
        self.ops[1].assert_not_called()

    def test_packed_launch_and_tail(self):
        self.run_reader(pack=True, T=13)  # G = 5: two groups, three tail rows
        # Operator args: (q, kv, indices, sm_scale, heads_per_group, topk_length, out)
        packed_call, tail_call = self.ops[0].call_args_list
        self.assertEqual(tuple(packed_call.args[0].shape), (2, 60, 576))
        self.assertEqual(packed_call.args[3:5], (0.05, 12))
        self.assertEqual(tuple(packed_call.args[5].shape), (2,))
        self.assertEqual(tuple(packed_call.args[6].shape), (2, 60, 512))
        self.assertEqual(tuple(tail_call.args[0].shape), (3, 12, 576))
        self.assertEqual(tail_call.args[4], 0)

    def test_batch_shapes_and_padding(self):
        cases = (
            dict(T=10, split_q=False),  # no tail, q as one [T, H, 576] tensor
            dict(T=3),  # fewer tokens than a group: the tail launch only
            dict(T=11, valid=[128, 64, 1, 0, 37]),  # -1 padded top-k rows
            dict(T=13, pad_rows=7),  # DP padding: more top-k rows than tokens
        )
        for kw in cases:
            for pack in (True, False):
                with self.subTest(pack=pack, **kw):
                    self.run_reader(pack=pack, **kw)
                    for call in self.ops[0].call_args_list:
                        self.assertEqual(call.args[0].shape[0], call.args[2].shape[0])

    def test_head_counts(self):
        for h, group in ((1, 5), (8, 5), (16, 4), (32, 2)):
            self.ops[0].reset_mock()
            self.run_reader(pack=True, T=2 * group + 1, h=h)
            self.assertEqual(self.ops[0].call_args_list[0].args[0].shape[1], group * h)

    def test_wide_top_k_runs_unpacked(self):
        self.run_reader(pack=True, T=5, h=12, topk=2048, pages=40)  # 5 x 2048 ids fit
        self.assertEqual(reader.reader_stats()["packed"], 1)
        with patch.object(reader, "_MAX_GROUP", 7):  # 7 x 2048 exceed the packed row
            self.run_reader(pack=True, T=7, h=8, topk=2048, pages=40)
        self.assertEqual(reader.reader_stats()["unpacked"], 1)

    def test_reader_follows_the_measured_overlap(self):
        cases = (
            # No shared keys: unpacked for `hold` calls, then packing is probed again.
            (dict(pages=40, disjoint=True), [(1, 0), (1, 1), (1, 2), (2, 2), (2, 3)]),
            # 192 rows, 128 ids per token: heavy overlap, packing stays on.
            (dict(pages=3), [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]),
        )
        for kw, expected in cases:
            reader._reset_for_tests()
            reader._GUARD = reader._OverlapGuard(0.8, probe_every=1, hold=2)
            kinds = []
            for _ in expected:
                self.run_reader(pack=True, T=10, **kw)
                stats = reader.reader_stats()
                kinds.append((stats["packed"], stats["unpacked"]))
            self.assertEqual(kinds, expected)

    def test_declined_calls(self):
        q, q_rope, kv_cache, indices, q_all = make_inputs()
        covered = dict(q=q, q_rope=q_rope, kv_cache=kv_cache, indices=indices)
        wide = dict(q=q.repeat(1, 4, 1), q_rope=q_rope.repeat(1, 4, 1))
        bad_width = "top-k width is not a multiple of 64 in 128..2048"
        too_large = "BF16 copy of the gathered rows above the size limit"
        cases = (
            ("q is not a 3-d bf16 tensor", dict(q=q.float())),
            ("empty batch", dict(q=q[:0], q_rope=q_rope[:0])),
            ("head count outside 1..32", wide),
            ("query width is not 512 + 64", dict(q=q_all)),
            ("q_rope does not match q", dict(q_rope=q_rope[:, :6])),
            ("KV cache is not FP8", dict(kv_cache=kv_cache.to(torch.bfloat16))),
            ("KV buffer is not contiguous V32 pages", dict(kv_cache=kv_cache[::2])),
            ("top-k ids are not contiguous int32 rows", dict(indices=indices[:, ::2])),
            ("fewer top-k rows than query tokens", dict(indices=indices[:5])),
            (bad_width, dict(indices=indices[:, :64].contiguous())),
            (too_large, dict(max_bf16_bytes=4 * 64 * 1152 - 1)),
        )
        for why, change in cases:
            with self.subTest(why=why):
                args = {**covered, **change}
                self.assertIsNone(reader.k31_prefill_reader(sm_scale=0.05, **args))
                self.assertIn(why, reader.reader_stats()["declined_reasons"])
        self.assertEqual(reader.reader_stats()["declined"], len(cases))
        for op in self.ops:
            op.assert_not_called()

    def test_runtime_error_disables_the_reader(self):
        self.ops[0].side_effect = RuntimeError("CUDA out of memory")
        q, q_rope, kv_cache, indices, _ = make_inputs()
        with self.assertLogs(reader.logger, level="ERROR"):
            for _ in range(2):
                out = reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05)
                self.assertIsNone(out)
        self.assertEqual(self.ops[0].call_count, 1)  # the second call stops earlier
        self.assertIn("out of memory", reader.reader_stats()["disabled"])

    def test_missing_operators_or_wheel_keep_stock_with_one_warning(self):
        # An sgl-flash-mla before the token-packed prefill fails the capability check.
        old_wheel = SimpleNamespace(supports_sparse_prefill_token_packing=lambda: False)
        cases = (
            (dict(return_value=old_wheel), "flash_mla_sparse_fwd_packed"),
            (dict(side_effect=ImportError("No module named 'flash_mla'")), "No module"),
        )
        for how, text in cases:
            reader._reset_for_tests()
            with self.subTest(text=text), patch.object(reader, "_flash_mla", **how):
                with self.assertLogs(reader.logger, level="WARNING") as logs:
                    self.assertFalse(reader.k31_prefill_reader_available())
                    self.assertFalse(reader.k31_prefill_reader_available())
                self.assertEqual(len(logs.output), 1)
                self.assertIn("unavailable", logs.output[0])
                self.assertIn(text, logs.output[0])


_COVERED = dict(requested="packed", deterministic=False, is_cuda=True, is_sm100=True)
_COVERED.update(shards_index_k=True, prefill_cp=False, index_kpool=1, hisparse=False)
_COVERED.update(kv_cache_store_fp8=True, kv_format="V32", physical_page_size=64)
_COVERED.update(local_num_q_heads=12, prefill_impl="flashmla_kv")


def _decide(available=lambda: True, **changes):
    args = {**_COVERED, "available": available, **changes}
    mode, why = _k31_sparse_prefill_reader_decision(**args)
    assert mode == _resolve_k31_sparse_prefill_reader(**args)
    return mode, why


class TestEligibility(CustomTestCase):
    def test_flag_is_off_by_default_and_off_never_asks_for_the_operators(self):
        fields = {f.name: f for f in msgspec.structs.fields(ExecKernel)}
        self.assertEqual(fields["k31_sparse_prefill_reader"].default, "off")
        available = MagicMock(side_effect=AssertionError)
        self.assertEqual(_decide(requested="off", available=available)[0], "off")

    def test_active_decisions(self):
        self.assertEqual(_decide(), ("packed", "all requirements met"))
        self.assertEqual(_decide(requested="unpacked")[0], "unpacked")
        mode, why = _decide(deterministic=True)
        self.assertEqual(mode, "unpacked")
        self.assertIn("deterministic", why)

    def test_uncovered_deployments_stay_off_and_say_why(self):
        available = MagicMock(return_value=True)
        cases = (
            (dict(is_cuda=False), "CUDA"),
            (dict(is_sm100=False), "SM100"),
            # Kimi-K3.1 defaults to prims_ts on SM100-family GPUs: another KV layout,
            # and the extend never reaches the flashmla_kv branch the reader hooks.
            (dict(prefill_impl="prims_ts"), "the reader needs 'flashmla_kv'"),
            (dict(prefill_impl=None), "--dsa-prefill-backend is None"),
            (dict(shards_index_k=False), "index-K"),
            (dict(prefill_cp=True), "prefill CP"),
            (dict(index_kpool=2), "K-pool is 2"),
            (dict(kv_cache_store_fp8=False), "fp8_ds_mla"),
            (dict(kv_format="V32_NO_ROPE"), "V32_NO_ROPE"),
            (dict(physical_page_size=32), "page size is 32"),
            (dict(local_num_q_heads=48), "48 heads"),
            (dict(hisparse=True), "HiSparse"),
        )
        for change, text in cases:
            with self.subTest(change=change):
                mode, why = _decide(available=available, **change)
                self.assertEqual(mode, "off")
                self.assertIn(text, why)
        available.assert_not_called()  # the operators are looked for last
        mode, why = _decide(available=lambda: False)
        self.assertEqual(mode, "off")
        self.assertIn("sgl-flash-mla", why)
        _, why = _decide(prefill_impl="trtllm", kv_cache_store_fp8=False)
        self.assertIn("--dsa-prefill-backend", why)
        self.assertIn("fp8_ds_mla", why)  # every failed requirement is listed

    def test_constructor_passes_the_backend_and_logs_the_reason(self):
        source = inspect.getsource(DeepseekSparseAttnBackend.__init__)
        self.assertIn("requested %s, active %s (%s)", source)
        self.assertIn("_k31_sparse_prefill_reader_decision(", source)
        self.assertIn("prefill_impl=get_exec().kernel.dsa_prefill_backend", source)


class TestDispatch(CustomTestCase):
    def _call(self, mode, forward_mode=ForwardMode.EXTEND, v_head_dim=512, q_rope="r"):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.k31_sparse_prefill_reader = mode
        backend.physical_page_size = 64
        backend.kv_cache_dim = 656
        layer = SimpleNamespace(v_head_dim=v_head_dim, scaling=0.05)
        batch = SimpleNamespace(forward_mode=forward_mode)
        kv = torch.zeros(2 * 64, 656, dtype=torch.uint8)
        with patch.object(reader, "k31_prefill_reader", return_value="reader") as fn:
            out = backend._k31_prefill_reader_extend(
                layer, batch, "q_nope", q_rope, "q_all", kv, "table"
            )
        return out, fn

    def test_off_and_uncovered_calls_keep_stock(self):
        cases = [dict(mode="off"), dict(mode="packed", v_head_dim=256)]
        for forward_mode in ForwardMode:  # speculative extends and every other mode
            if not forward_mode.is_extend_without_speculative():
                cases.append(dict(mode="packed", forward_mode=forward_mode))
        for kw in cases:
            with self.subTest(**kw):
                out, fn = self._call(**kw)
                self.assertIsNone(out)
                fn.assert_not_called()

    def test_extend_goes_to_the_reader_with_the_gathered_buffer(self):
        out, fn = self._call("packed")
        self.assertEqual(out, "reader")  # a decline (None) is handed back the same way
        q, q_rope, kv, table, scale = fn.call_args.args
        self.assertEqual((q, q_rope, table, scale), ("q_nope", "r", "table", 0.05))
        self.assertEqual(tuple(kv.shape), (2, 64, 1, 656))
        self.assertTrue(fn.call_args.kwargs["pack"])
        out, fn = self._call("unpacked", q_rope=None)
        self.assertEqual(fn.call_args.args[0], "q_all")
        self.assertFalse(fn.call_args.kwargs["pack"])

    def test_reader_is_tried_before_the_stock_token_packing(self):
        source = inspect.getsource(DeepseekSparseAttnBackend.forward_extend)
        branch = source[source.index('dsa_impl == "flashmla_kv"') :]
        stock = branch.index("self.flashmla_prefill_token_packing")
        self.assertLess(branch.index("_k31_prefill_reader_extend"), stock)
        self.assertLess(branch.index("if reader_out is not None"), stock)


if __name__ == "__main__":
    unittest.main()
