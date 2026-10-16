"""Noncontiguous CP queries must keep their original causal and SWA positions."""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp import base, interleave
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDenseInterleave(CustomTestCase):
    def test_odd_requests_with_prefix_and_padding(self):
        # A suffix-shaped causal mask is wrong for every-fourth-token Q shards.
        # The independent reference attends to each original request prefix.
        torch.manual_seed(7)
        extend, prefix = [5, 3, 7], [3, 0, 2]
        for kv_heads in (1, 2, 4):
            for window in (-1, 2):
                with self.subTest(kv_heads=kv_heads, window=window):
                    keys = [
                        torch.randn(p + n, kv_heads, 8) for p, n in zip(prefix, extend)
                    ]
                    values = [
                        torch.randn(p + n, kv_heads, 6) for p, n in zip(prefix, extend)
                    ]
                    queries = torch.randn(sum(extend), 4, 8)

                    def attend(q, request, length):
                        start = 0 if window < 0 else max(0, length - 1 - window)
                        k = keys[request][start:length].repeat_interleave(
                            4 // kv_heads, 1
                        )
                        v = values[request][start:length].repeat_interleave(
                            4 // kv_heads, 1
                        )
                        scores = torch.einsum("hd,shd->hs", q, k) / 8**0.5
                        return torch.einsum("hs,shd->hd", scores.softmax(-1), v)

                    expected, offset = [], 0
                    for request, (p, n) in enumerate(zip(prefix, extend)):
                        for pos in range(n):
                            expected.append(
                                attend(queries[offset + pos], request, p + pos + 1)
                            )
                        offset += n
                    expected = torch.stack(expected)
                    for rank in range(4):
                        parallel = SimpleNamespace(attn_cp_rank=rank)
                        with patch.object(base, "get_parallel", return_value=parallel):
                            strategy = interleave.InterleaveCPStrategy(4)
                            batch = SimpleNamespace(
                                input_ids=torch.arange(sum(extend)),
                                extend_seq_lens_cpu=extend,
                                extend_seq_lens=torch.tensor(extend, dtype=torch.int32),
                                extend_prefix_lens=torch.tensor(
                                    prefix, dtype=torch.int32
                                ),
                                attn_cp_metadata=strategy.build_metadata(
                                    sum(extend), None, extend
                                ),
                            )
                            indices = strategy.local_q_indices(sum(extend), batch)
                            local_q = torch.cat(
                                [queries[indices], torch.zeros(2, 4, 8)]
                            )

                            def paged_attention(
                                q, cu_q, lengths, max_q, *, request_indices
                            ):
                                self.assertEqual(max_q, 1)
                                self.assertEqual(cu_q.tolist(), list(range(len(q) + 1)))
                                return torch.stack(
                                    [
                                        attend(row, int(req), int(length))
                                        for row, req, length in zip(
                                            q, request_indices, lengths
                                        )
                                    ]
                                )

                            actual = strategy.run_attention(
                                local_q, batch, "cpu", paged_attention
                            )
                            self.assertIsNotNone(
                                actual, "dense interleave attention was not dispatched"
                            )
                            torch.testing.assert_close(
                                actual[: len(indices)], expected[indices]
                            )
                            torch.testing.assert_close(
                                actual[len(indices) :], torch.zeros(2, 4, 6)
                            )

    def test_gathered_cache_rows_and_mla_return_contract(self):
        from sglang.srt.model_executor import forward_context

        # The collective boundary supplies rank-packed data. Real gather code
        # must strip sentinel padding and restore slots before the pool write.
        k = torch.arange(15 * 3).reshape(15, 1, 3).float()
        v = -torch.arange(15 * 2).reshape(15, 1, 2).float()
        layer = SimpleNamespace(is_cross_attention=False, k_scale=None, v_scale=None)
        for rank in range(4):
            parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_group=None)
            with patch.object(base, "get_parallel", return_value=parallel):
                strategy = interleave.InterleaveCPStrategy(4)
                meta = strategy.build_metadata(15, None, [5, 3, 7])
                meta.per_rank_logical_token = list(meta.per_rank_actual_token)
                meta.per_rank_actual_token = [8] * 4
                batch = SimpleNamespace(
                    attn_cp_metadata=meta,
                    out_cache_loc=torch.arange(15).flip(0),
                    out_cache_loc_is_physical=True,
                )
                stored_k, stored_v = torch.zeros_like(k), torch.zeros_like(v)

                def write(layer, loc, keys, values, *scales):
                    self.assertTrue(loc.physical)
                    torch.testing.assert_close(loc.swa_loc, torch.arange(15))
                    stored_k[loc.loc] = keys
                    stored_v[loc.loc] = values

                def gather(output, local):
                    full = torch.cat([k, v], -1).reshape(15, *local.shape[1:])
                    for source in range(4):
                        shard = full[source::4]
                        output[source * 8 : (source + 1) * 8].fill_(12345)
                        output[source * 8 : source * 8 + len(shard)] = shard
                    torch.testing.assert_close(local[: len(k[rank::4])], full[rank::4])

                pool = SimpleNamespace(set_kv_buffer=write)
                with (
                    patch.object(interleave, "get_parallel", return_value=parallel),
                    patch.object(
                        interleave, "use_symmetric_memory", return_value=nullcontext()
                    ),
                    patch.object(
                        interleave, "is_allocation_symmetric", return_value=False
                    ),
                    patch.object(
                        interleave, "attn_cp_all_gather_into_tensor", side_effect=gather
                    ),
                    patch.object(
                        forward_context, "get_token_to_kv_pool", return_value=pool
                    ),
                    patch.object(torch.cuda, "current_stream", return_value=None),
                ):
                    strategy.materialize_full_kv(
                        batch, layer, k[rank::4], v[rank::4], swa_loc=torch.arange(17)
                    )
                    torch.testing.assert_close(stored_k.flip(0), k)
                    torch.testing.assert_close(stored_v.flip(0), v)
                    # DSA expects a tuple, never a cache-write side effect.
                    full_k, full_rope = strategy.materialize_full_mla_kv(
                        batch, layer, k[rank::4], v[rank::4]
                    )
                    torch.testing.assert_close(full_k, k)
                    torch.testing.assert_close(full_rope, v)


class TestDenseInterleaveBackend(CustomTestCase):
    @contextmanager
    def backend_case(self, *, mla=False, version=3, zigzag_layout=False):
        from sglang.srt.layers.attention import flashattention_backend as fa
        from sglang.srt.layers.cp import zigzag
        from sglang.srt.model_executor import forward_context

        torch.manual_seed(31)
        strategy = (
            zigzag.ZigzagCPStrategy
            if zigzag_layout
            else interleave.InterleaveCPStrategy
        )(2)
        parallel = SimpleNamespace(attn_cp_rank=1, attn_cp_group=None)
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            attn_type=fa.AttentionType.DECODER,
            sliding_window_size=-1,
            tp_q_head_num=2,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=5 if mla else 4,
            v_head_dim=3,
            scaling=0.5,
            logit_cap=0.0,
            k_scale=torch.tensor(2.0),
            v_scale=torch.tensor(3.0),
        )
        page_table = torch.randperm(20).reshape(2, 10).int()
        key = torch.randn(20, 1, 2 if mla else 4)
        value = torch.randn(20, 1, 3)
        q = torch.randn(12, 2, layer.head_dim)
        loc = torch.cat([page_table[0, 2:7], page_table[1, 1:8]]).long()
        batch = SimpleNamespace(
            input_ids=torch.arange(12),
            batch_size=2,
            extend_seq_lens_cpu=[5, 7],
            extend_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([2, 1], dtype=torch.int32),
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            attn_attend_prefix_cache=None,
            out_cache_loc=loc,
            out_cache_loc_is_physical=True,
        )
        writes, calls, gathers = [], [], []
        stored_k, stored_v = key.clone(), value.clone()
        stored_k[loc] = 0
        stored_v[loc] = 0

        def write(layer, target, k, v, *scales):
            writes.append(target)
            self.assertTrue(target.physical)
            torch.testing.assert_close(target.loc, loc)
            stored_k[target.loc], stored_v[target.loc] = (v, k) if mla else (k, v)

        pool = SimpleNamespace(
            set_kv_buffer=write,
            set_mla_kv_buffer=write,
            get_kv_buffer=lambda layer_id: (stored_k, stored_v),
            get_key_buffer=lambda layer_id: torch.cat([stored_v, stored_k], -1),
        )
        backend = object.__new__(fa.FlashAttentionBackend)
        backend.__dict__.update(
            use_mla=mla,
            fa_impl_ver=version,
            fa_skip_kv_cache=False,
            local_attn_builder=None,
            kv_cache_is_mxfp8=False,
            use_sliding_window_kv_pool=False,
            token_to_kv_pool=pool,
            kv_cache_dtype_str="fp8",
            kv_cache_dtype=torch.float32,
            page_size=1,
            num_splits=1,
            device="cpu",
            topk=1,
            forward_metadata=SimpleNamespace(
                page_table=page_table,
                cu_seqlens_q=torch.tensor([0, 5, 12], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([7, 8], dtype=torch.int32),
                max_seq_len_q=7,
                cu_seqlens_k=torch.tensor([0, 7, 15], dtype=torch.int32),
                swa_spec_metadata=None,
                swa_out_cache_loc=None,
            ),
        )

        def kernel(**kw):
            calls.append(kw)
            result = []
            for row in range(len(kw["cache_seqlens"])):
                begin, end = map(int, kw["cu_seqlens_q"][row : row + 2])
                length = int(kw["cache_seqlens"][row])
                for i in range(begin, end):
                    slots = kw["page_table"][row, : length - (end - i - 1)].long()
                    keys = kw["k_cache"][slots, 0].repeat_interleave(2, 1)
                    vals = kw["v_cache"][slots, 0].repeat_interleave(2, 1)
                    kd = kw.get("k_descale")
                    vd = kw.get("v_descale")
                    if kd is not None:
                        self.assertEqual(kd.shape, (len(kw["cache_seqlens"]), 1))
                        keys = keys * kd[row]
                        vals = vals * vd[row]
                    score = torch.einsum("hd,shd->hs", kw["q"][i], keys)
                    if "qv" in kw:
                        score += torch.einsum("hd,shd->hs", kw["qv"][i], vals)
                    result.append(
                        torch.einsum(
                            "hs,shd->hd",
                            (score * kw["softmax_scale"]).softmax(-1),
                            vals,
                        )
                    )
            return torch.stack(result)

        with ExitStack() as stack:
            for module in (base, interleave, zigzag):
                stack.enter_context(
                    patch.object(module, "get_parallel", return_value=parallel)
                )
            stack.enter_context(
                patch.object(
                    zigzag, "get_device", return_value=SimpleNamespace(device="cpu")
                )
            )
            batch.attn_cp_metadata = strategy.build_metadata(12, [7, 8], [5, 7])
            indices = strategy.local_q_indices(12, batch)
            local_q = torch.cat([q[indices], torch.zeros(2, 2, layer.head_dim)])
            full_payload = (
                torch.cat([value[loc], key[loc]], -1)
                if mla
                else torch.cat([key[loc], value[loc]], -1)
            )

            def gather(output, local):
                gathers.append(local.shape)
                # The collective boundary supplies both real rank payloads.
                full = full_payload.reshape(12, *local.shape[1:])
                for rank in range(2):
                    parallel.attn_cp_rank = rank
                    rank_batch = SimpleNamespace(
                        input_ids=batch.input_ids,
                        attn_cp_metadata=strategy.build_metadata(12, [7, 8], [5, 7]),
                    )
                    rank_idx = strategy.local_q_indices(12, rank_batch)
                    target = output[rank * len(local) : (rank + 1) * len(local)]
                    target.zero_()
                    target[: len(rank_idx)] = full[rank_idx]
                parallel.attn_cp_rank = 1

            parallel.attn_cp_group = SimpleNamespace(all_gather_into_tensor=gather)
            stack.enter_context(
                patch.object(
                    interleave, "attn_cp_all_gather_into_tensor", side_effect=gather
                )
            )
            stack.enter_context(
                patch.object(
                    interleave, "use_symmetric_memory", return_value=nullcontext()
                )
            )
            stack.enter_context(
                patch.object(interleave, "is_allocation_symmetric", return_value=False)
            )
            stack.enter_context(
                patch.object(torch.cuda, "current_stream", return_value=None)
            )
            stack.enter_context(
                patch.object(forward_context, "get_token_to_kv_pool", return_value=pool)
            )
            stack.enter_context(
                patch.object(zigzag, "get_token_to_kv_pool", return_value=pool)
            )
            stack.enter_context(
                patch.object(fa, "get_cp_strategy", return_value=strategy)
            )
            stack.enter_context(patch.object(fa, "is_cp_active", return_value=True))
            stack.enter_context(
                patch.object(fa, "is_interleave", return_value=not zigzag_layout)
            )
            stack.enter_context(
                patch.object(fa, "flash_attn_with_kvcache", side_effect=kernel)
            )
            yield SimpleNamespace(
                backend=backend,
                layer=layer,
                batch=batch,
                q=local_q,
                k=(value if mla else key)[loc][indices],
                v=value[loc][indices],
                rope=key[loc][indices],
                calls=calls,
                writes=writes,
                gathers=gathers,
                full_q=q,
                key=key,
                value=value,
                indices=indices,
                page_table=page_table,
                stored_k=stored_k,
                stored_v=stored_v,
            )

    def test_forward_extend_page_rows_scales_and_mla_write(self):
        for mla, version, separate_rope in (
            (False, 3, False),
            (False, 4, False),
            (True, 3, False),
            (True, 3, True),
            (True, 4, True),
        ):
            with (
                self.subTest(mla=mla, version=version, separate_rope=separate_rope),
                self.backend_case(mla=mla, version=version) as c,
            ):
                actual = c.backend.forward_extend(
                    c.q[:, :, :3] if separate_rope else c.q,
                    c.k,
                    c.v,
                    c.layer,
                    c.batch,
                    q_rope=c.q[:, :, 3:] if separate_rope else None,
                    k_rope=c.rope if mla else None,
                )
                self.assertEqual(len(c.writes), 1)
                torch.testing.assert_close(c.stored_k, c.key)
                torch.testing.assert_close(c.stored_v, c.value)
                call = c.calls[0]
                torch.testing.assert_close(
                    call["page_table"], c.page_table[[0, 0, 1, 1, 1, 1]]
                )
                self.assertEqual(call["cache_seqlens"].tolist(), [4, 6, 2, 4, 6, 8])
                self.assertIsNone(call["cu_seqlens_k_new"])
                if mla:
                    torch.testing.assert_close(call["qv"], c.q[:6, :, :3])
                    torch.testing.assert_close(call["q"], c.q[:6, :, 3:])
                expected = []
                for global_idx in c.indices.tolist():
                    request, position = (
                        (0, global_idx + 2)
                        if global_idx < 5
                        else (1, global_idx - 5 + 1)
                    )
                    slots = c.page_table[request, : position + 1].long()
                    k = c.key[slots].repeat_interleave(2, 1) * (
                        2 if version == 3 else 1
                    )
                    v = c.value[slots].repeat_interleave(2, 1) * (
                        3 if version == 3 else 1
                    )
                    query = c.full_q[global_idx]
                    score = torch.einsum(
                        "hd,shd->hs", query[:, 3:] if mla else query, k
                    )
                    if mla:
                        score += torch.einsum("hd,shd->hs", query[:, :3], v)
                    expected.append(
                        torch.einsum("hs,shd->hd", (score * 0.5).softmax(-1), v)
                    )
                expected = torch.cat(
                    [torch.stack(expected), torch.zeros(2, 2, 3)]
                ).flatten(1)
                torch.testing.assert_close(actual, expected)

    def test_fa4_mla_rejects_before_collective_or_cache_write(self):
        for option, value in (
            ("sliding_window_size", 4),
            ("logit_cap", 1.0),
            ("num_splits", 2),
        ):
            with (
                self.subTest(option=option),
                self.backend_case(mla=True, version=4) as c,
            ):
                setattr(c.backend if option == "num_splits" else c.layer, option, value)
                before = c.stored_k.clone()
                with self.assertRaisesRegex(NotImplementedError, "FA4.*MLA"):
                    c.backend.forward_extend(
                        c.q, c.k, c.v, c.layer, c.batch, k_rope=c.rope
                    )
                self.assertEqual(c.gathers, [])
                self.assertEqual(c.writes, [])
                self.assertEqual(c.calls, [])
                torch.testing.assert_close(c.stored_k, before)

    def test_zigzag_callback_geometry_and_single_mla_write(self):
        for mla in (False, True):
            with (
                self.subTest(mla=mla),
                self.backend_case(mla=mla, zigzag_layout=True) as c,
            ):
                c.backend.forward_extend(
                    c.q, c.k, c.v, c.layer, c.batch, k_rope=c.rope if mla else None
                )
                self.assertEqual(len(c.writes), 1)
                meta = c.batch.attn_cp_metadata
                self.assertEqual(len(c.calls), 2)
                for call, suffix in zip(c.calls, ("prev", "next")):
                    self.assertIs(call["page_table"], c.page_table)
                    self.assertIs(
                        call["cu_seqlens_q"],
                        getattr(meta, f"cu_seqlens_q_{suffix}_tensor"),
                    )
                    self.assertIs(
                        call["cache_seqlens"], getattr(meta, f"kv_len_{suffix}_tensor")
                    )
                    self.assertIs(
                        call["cu_seqlens_k_new"],
                        c.backend.forward_metadata.cu_seqlens_k,
                    )
                    self.assertEqual(call["window_size"], (-1, -1))
                torch.testing.assert_close(c.stored_k, c.key)
                torch.testing.assert_close(c.stored_v, c.value)


class TestQwen2ContextParallelLayout(CustomTestCase):
    def test_head_projection_and_cache_layout_follow_attention_tp(self):
        from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
        from sglang.srt.models import qwen2
        from sglang.srt.runtime_context import SpawnRanks, reset_context
        from sglang.srt.server_args import ServerArgs
        from sglang.test.parallel_groups import parallel_scope, publish, rank_size

        reset_context()
        self.addCleanup(reset_context)
        server = ServerArgs(model_path="dummy", device="cpu", tp_size=2)
        publish(server, role="test", ranks=SpawnRanks(world_rank=1))
        initialize_dp_attention_flags(server)
        # CP2 holds full attention heads; CP-off retains ordinary TP2 shards.
        for cp_size in (2, 1):
            with (
                self.subTest(cp_size=cp_size),
                parallel_scope(
                    tp_rank=1,
                    tp_size=2,
                    attn_tp_rank=0 if cp_size == 2 else 1,
                    attn_tp_size=2 // cp_size,
                    attn_cp_size=cp_size,
                    attn_cp_rank=1 if cp_size == 2 else 0,
                    pp_group=SimpleNamespace(
                        is_first_rank=False,
                        is_last_rank=False,
                        rank_in_group=0,
                        world_size=1,
                    ),
                ),
                patch.object(qwen2, "get_rope", return_value=torch.nn.Identity()),
            ):
                attention = qwen2.Qwen2Attention(32, 4, 4, head_dim=8)
                heads = 4 // (2 // cp_size)
                self.assertEqual(
                    (attention.num_heads, attention.num_kv_heads), (heads, heads)
                )
                placement = (0, 1) if cp_size == 2 else (1, 2)
                self.assertEqual(rank_size(attention.qkv_proj), placement)
                self.assertEqual(rank_size(attention.o_proj), placement)
                self.assertEqual(attention.qkv_proj.weight.shape, (3 * heads * 8, 32))
                self.assertEqual(attention.o_proj.weight.shape, (32, heads * 8))
                self.assertEqual(attention.attn.tp_k_head_num, heads)
                with patch.object(
                    qwen2, "make_pp_layers", return_value=(torch.nn.ModuleList(), 0, 0)
                ):
                    model = qwen2.Qwen2Model(
                        SimpleNamespace(vocab_size=32, num_hidden_layers=1)
                    )
                self.assertEqual(model._kv_cache_parallel_layout, placement)


if __name__ == "__main__":
    unittest.main()
