"""FlashInfer's kernel over a sliding-window layer's window read, for a target
verify and a draft extend, against attention computed from positions alone.

A verify tree's rows sit at their tree depth, so the window is folded into the
tree's mask and the kernel is planned without one; a draft extend's queries
are consecutive positions, so the kernel is planned with the layer's window.
Each reads only the window, from its first token on.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import flashinfer_backend as fb
from sglang.srt.mem_cache.kv.alloc.id_space import window_read
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

HEADS, KV_HEADS, DIM, WINDOW = 32, 4, 128, 128


def _branched():
    parents = [-1, 0] + [1] * 62
    tree = torch.zeros(len(parents), len(parents), dtype=torch.bool)
    for row, parent in enumerate(parents):
        tree[row, row] = True
        while parent >= 0:
            tree[row, parent] = True
            parent = parents[parent]
    return tree


def _reference(q, k, v, visible):
    k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
    v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / DIM**0.5
    scores = scores.masked_fill(~visible[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())


def _layer(window):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=HEADS,
        tp_k_head_num=KV_HEADS,
        tp_v_head_num=KV_HEADS,
        head_dim=DIM,
        sliding_window_size=window,
        scaling=DIM**-0.5,
        logit_cap=0,
        k_scale_float=None,
        v_scale_float=None,
        is_cross_attention=False,
        attn_type=fb.AttentionType.DECODER,
    )


def _backend(wrappers, kv_cache, *, use_ragged=False):
    backend = fb.FlashInferAttnBackend.__new__(fb.FlashInferAttnBackend)
    backend.num_wrappers = 2
    backend.dispatch_reason = fb.WrapperDispatch.SLIDING_WINDOW
    backend.prefill_uses_dequant_workspace = False
    backend.token_to_kv_pool = SimpleNamespace(
        layer_router=SimpleNamespace(
            read=lambda layer_id: SimpleNamespace(kv=lambda: kv_cache)
        )
    )
    backend.forward_metadata = fb.PrefillMetadata(wrappers, use_ragged, False)
    backend.is_dllm_model = False
    return backend


class TestFlashInferWindowForward(CustomTestCase):
    def test_forward_preserves_the_planned_window(self):
        """A mask-owned window must not become a row-indexed kernel window
        when the backend forwards a paged tree or a ragged prefix."""
        from flashinfer import (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
        )

        cases = (
            ("tree", False, WINDOW, True, False, True, -1),
            ("draft", False, WINDOW, False, False, True, WINDOW),
            ("narrower_layer", False, WINDOW // 2, False, False, True, WINDOW // 2),
            ("full", False, -1, False, False, True, -1),
            ("multi_item", False, WINDOW, False, True, True, -1),
            ("ragged_prefix", True, WINDOW, True, False, True, -1),
            ("ragged_full", True, -1, False, False, True, -1),
            ("single_window", False, WINDOW // 2, False, False, False, WINDOW // 2),
            ("single_multi_item", False, WINDOW // 2, False, True, False, -1),
        )
        for (
            name,
            ragged,
            layer_window,
            masked,
            multi_item,
            dispatch_window,
            expected,
        ) in cases:
            with self.subTest(name=name):
                wrapper = BatchPrefillWithPagedKVCacheWrapper.__new__(
                    BatchPrefillWithPagedKVCacheWrapper
                )
                # Replace the external planner/kernel, retaining FlashInfer's
                # real forward adapters and SGLang's planning/dispatch methods.
                wrapper.begin_forward = lambda *a, window_left=-1, **kw: setattr(
                    wrapper, "_window_left", window_left
                )
                wrapper.run = lambda q, *a, **kw: torch.full_like(
                    q, wrapper._window_left
                )
                wrapper.run_return_lse = lambda q, *a, **kw: (wrapper.run(q), None)
                backend = _backend(
                    [wrapper, None] if layer_window >= 0 else [None, wrapper],
                    (None, None),
                    use_ragged=ragged,
                )
                if not dispatch_window:
                    backend.num_wrappers = 1
                    backend.dispatch_reason = None
                    backend.forward_metadata.prefill_wrappers = [wrapper]
                multi = (
                    fb.MultiItemScoringParams(prefix_len_ptr=torch.zeros(1))
                    if multi_item
                    else None
                )
                backend.forward_metadata.multi_item_params = multi
                updater = fb.FlashInferIndicesUpdaterPrefill.__new__(
                    fb.FlashInferIndicesUpdaterPrefill
                )
                updater.attn_backend = SimpleNamespace(
                    runner_ids=SimpleNamespace(reads_are_translated=False),
                    dq_paged_kernel_lens=None,
                )
                updater.num_qo_heads, updater.num_kv_heads, updater.head_dim = (
                    HEADS,
                    KV_HEADS,
                    DIM,
                )
                updater.q_data_type = updater.data_type = torch.float32
                updater.kv_last_page_len = torch.ones(1, dtype=torch.int32)
                seq = torch.tensor([3], dtype=torch.int32)
                num_queries = 2 if dispatch_window else 1
                prefix = torch.tensor([3 - num_queries], dtype=torch.int32)
                paged_lens = prefix if ragged else seq
                num_keys = int(paged_lens.sum())
                read = window_read(
                    seq_lens=seq, prefix_lens=prefix, window=WINDOW, prefix_only=ragged
                )
                updater.call_begin_forward(
                    SimpleNamespace(begin_forward=lambda *a, **kw: None),
                    wrapper,
                    torch.tensor([0]),
                    paged_lens,
                    num_keys,
                    seq,
                    prefix,
                    None,
                    torch.zeros(2, dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                    ragged,
                    None,
                    custom_kv_indices=torch.arange(num_keys, dtype=torch.int32),
                    cross_attention_custom_mask=torch.ones(
                        num_queries * num_keys, dtype=torch.bool
                    )
                    if masked
                    else None,
                    multi_item_params=multi,
                    window=read if dispatch_window and layer_window >= 0 else None,
                    plan=None,
                )
                suffix = BatchPrefillWithRaggedKVCacheWrapper.__new__(
                    BatchPrefillWithRaggedKVCacheWrapper
                )
                suffix.run_return_lse = lambda q, *a, **kw: (
                    torch.full_like(q, suffix._window_left),
                    None,
                )
                backend.prefill_wrapper_ragged = suffix
                q = torch.zeros(num_queries, HEADS, DIM)
                kv = torch.zeros(num_queries, KV_HEADS, DIM)
                layer_windows = (
                    (layer_window, WINDOW // 2, 0, WINDOW)
                    if expected >= 0 and dispatch_window
                    else (layer_window,)
                )
                for current_window in layer_windows:
                    with patch.object(
                        fb, "_safe_merge_state", lambda o1, s1, o2, s2: (o1 + o2, None)
                    ):
                        out = backend.forward_extend(
                            q, kv, kv, _layer(current_window), None, save_kv_cache=False
                        )
                    effective_window = current_window if expected >= 0 else expected
                    expected_output = effective_window + (
                        current_window if ragged else 0
                    )
                    torch.testing.assert_close(
                        out, torch.full_like(out, expected_output)
                    )
                    self.assertEqual(wrapper._window_left, effective_window)

    def test_fast_plan_refreshes_the_runtime_window(self):
        """Replay planning must reset a stale forward override before the
        next paged kernel uses the captured mask."""
        ptr = torch.tensor([0, 2], dtype=torch.int32)
        indices = torch.arange(2, dtype=torch.int32)
        last = torch.ones(1, dtype=torch.int32)
        wrapper = SimpleNamespace(
            is_cuda_graph_enabled=True,
            _backend="fa2",
            _cached_module=SimpleNamespace(plan=lambda *a: None),
            _max_total_num_rows=None,
            _window_left=WINDOW,
            _qo_indptr_buf=ptr.clone(),
            _paged_kv_indptr_buf=ptr.clone(),
            _paged_kv_last_page_len_buf=last.clone(),
            _paged_kv_indices_buf=indices.clone(),
            _float_workspace_buffer=None,
            _int_workspace_buffer=None,
            _pin_memory_int_workspace_buffer=None,
            device=torch.device("cpu"),
        )
        fb.fast_prefill_plan(
            wrapper,
            ptr,
            ptr,
            indices,
            last,
            HEADS,
            KV_HEADS,
            DIM,
            1,
            window_left=-1,
            qo_indptr_host=ptr,
            kv_indptr_host=ptr,
            kv_lens_host=torch.tensor([2], dtype=torch.int32),
            max_q_len=2,
            max_kv_len=2,
        )
        self.assertEqual(wrapper._window_left, -1)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFlashInferWindowRead(CustomTestCase):
    def setUp(self):
        import flashinfer

        torch.manual_seed(0)
        self.flashinfer = flashinfer
        self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")

    def _run(self, seq_lens, width, read, mask, window_left, total_lens):
        """Plan and run over the window read of each request's KV, whose
        tokens sit at slots ``[base_i, base_i + total_lens_i)``."""
        bs = len(seq_lens)
        n = sum(total_lens)
        k = torch.randn(n, 1, KV_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        q = torch.randn(bs * width, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        bases = torch.tensor([0] + total_lens[:-1]).cumsum(0).tolist()
        lens = (read.lens + width).tolist()
        indices = torch.cat(
            [
                torch.arange(b + s, b + s + l)
                for b, s, l in zip(bases, read.start.tolist(), lens)
            ]
        ).to(torch.int32)
        kv_indptr = torch.tensor([0] + lens).cumsum(0).to(torch.int32)
        qo_indptr = torch.arange(0, (bs + 1) * width, width, dtype=torch.int32)
        wrapper = self.flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr.cuda(),
            kv_indptr.cuda(),
            indices.cuda(),
            torch.ones(bs, dtype=torch.int32, device="cuda"),
            HEADS,
            KV_HEADS,
            DIM,
            1,
            custom_mask=mask,
            causal=mask is None,
            window_left=window_left,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        backend = _backend([wrapper, None], (k, v))
        out = backend.forward_extend(
            q, None, None, _layer(WINDOW), None, save_kv_cache=False
        ).view_as(q)
        return q, k[:, 0], v[:, 0], bases, out

    def test_verify_tree_reads_its_window(self):
        """BUG REGRESSION. The verify read its window wrapper from column 0;
        and a row-indexed kernel window misplaces a branched tree's rows."""
        tree = _branched().cuda()
        width = tree.shape[0]
        seq_lens = [10, 256, 129]
        seq = torch.tensor(seq_lens, device="cuda")
        read = window_read(
            seq_lens=seq, prefix_lens=seq, window=WINDOW, prefix_only=False
        )
        full = torch.cat(
            [
                torch.cat(
                    [torch.ones(width, s, dtype=torch.bool, device="cuda"), tree], 1
                ).flatten()
                for s in seq_lens
            ]
        )
        mask = read.tree_mask(full, width=width, read_lens_sum=int(read.lens.sum()))
        total = [s + width for s in seq_lens]
        q, k, v, bases, out = self._run(seq_lens, width, read, mask, -1, total)
        depth = tree.sum(-1) - 1
        for i, s in enumerate(seq_lens):
            keys = torch.arange(s + width, device="cuda")
            pos = s + depth
            prefix = (keys[None, :] < s) & (keys[None, :] >= pos[:, None] - WINDOW)
            visible = prefix.clone()
            visible[:, s:] = tree
            sl = slice(bases[i], bases[i] + s + width)
            expected = _reference(q[i * width : (i + 1) * width], k[sl], v[sl], visible)
            torch.testing.assert_close(
                out[i * width : (i + 1) * width].float(),
                expected,
                atol=2e-2,
                rtol=2e-2,
            )

    def test_draft_extend_reads_its_window(self):
        """A draft extend's queries are its sequence's last tokens; planned
        with the window, each sees the keys from its position minus the
        window on."""
        width = 3
        seq_lens = [5, 256, 140]
        seq = torch.tensor(seq_lens, device="cuda")
        read = window_read(
            seq_lens=seq, prefix_lens=seq - width, window=WINDOW, prefix_only=False
        )
        # The read already counts the queries' own keys.
        read_prefix = type(read)(
            start=read.start,
            lens=read.lens - width,
            window=read.window,
            window_left=read.window_left,
        )
        q, k, v, bases, out = self._run(
            seq_lens, width, read_prefix, None, read.window_left, seq_lens
        )
        for i, s in enumerate(seq_lens):
            keys = torch.arange(s, device="cuda")
            pos = torch.arange(s - width, s, device="cuda")
            visible = (keys[None, :] <= pos[:, None]) & (
                keys[None, :] >= pos[:, None] - WINDOW
            )
            sl = slice(bases[i], bases[i] + s)
            expected = _reference(q[i * width : (i + 1) * width], k[sl], v[sl], visible)
            torch.testing.assert_close(
                out[i * width : (i + 1) * width].float(),
                expected,
                atol=2e-2,
                rtol=2e-2,
            )


if __name__ == "__main__":
    unittest.main()
