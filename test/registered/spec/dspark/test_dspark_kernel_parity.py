"""Seeded triton-vs-torch parity sweep for the DSpark kernels.

Guards against toolchain drift (triton/torch upgrades) silently diverging
the triton implementations from their torch references. Every method calls
the production kernel pair directly via `Cls.torch(...)` / `Cls.triton(...)`
(no env-var dispatch) on a small set of adversarial inputs and compares
exactly, or with the tolerance the kernel is specified to meet.
"""

import types
import unittest

import torch

from sglang.kernels.ops.attention import (
    dsv4_attn_metadata_kernels as attn_metadata_kernels,
)
from sglang.kernels.ops.speculative import ragged_verify_kernels
from sglang.kernels.ops.speculative.dspark import (
    dspark_accept,
    dspark_attn_metadata,
    dspark_draft_model,
    dspark_schedule,
    dspark_verify_window,
)
from sglang.kernels.ops.speculative.dspark.markov_candidates import (
    markov_candidate_step,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
VOCAB = 129280


def _ri(lo, hi, shape, dtype=torch.int64, g=None):
    return torch.randint(lo, hi, shape, device=DEVICE, dtype=dtype, generator=g)


def _layout(verify_lens, graph_num_tokens):
    return RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
    )


class _Bf16Linear(torch.nn.Module):
    quant_method = None

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


def _case_accept_greedy(tc):
    torch.manual_seed(0)
    bs, t = 8, 6
    candidates = _ri(0, 200, (bs, t))
    target_logits = torch.randn(bs * t, 200, device=DEVICE)
    for cutoff in (None, _ri(1, t + 1, (bs,), torch.int32)):
        tc._parity(
            dspark_accept.AcceptGreedy,
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=t,
            cutoff_verify_lens=cutoff,
        )
    # gather_row_bonus: bonus token at a per-row column index.
    table, idx = _ri(0, VOCAB, (64, t)), _ri(0, t, (64,), torch.int32)
    ref = table[torch.arange(64, device=DEVICE), idx.long()]
    tc._eq(dspark_accept.gather_row_bonus_triton(table=table, idx=idx), ref)


def _case_accept_sampling(tc):
    torch.manual_seed(1)
    bs, t = 64, 6
    accept_index = _ri(0, bs * t, (bs, t))
    predicts = _ri(0, VOCAB, (bs * t,))
    correct_len = _ri(0, t, (bs,), torch.int32)
    rows = torch.arange(bs, device=DEVICE)
    ref = predicts[accept_index[rows, correct_len.long()].long()]
    got = dspark_accept.gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=correct_len
    )
    tc._eq(got, ref)


def _case_build_block_seq_lens_causal(tc):
    torch.manual_seed(2)
    seq_lens = _ri(1, 100000, (128,))
    for block_size in (1, 5, 7):
        tc._parity(
            dspark_attn_metadata.BuildBlockSeqLensCausal,
            seq_lens=seq_lens,
            block_size=block_size,
            device=DEVICE,
        )


def _case_build_out_tokens(tc):
    torch.manual_seed(3)
    bs, gamma = 64, 5
    for cl_dtype in (torch.int32, torch.int64):
        # Bonus insertion swept through every position 0..gamma.
        cl = (torch.arange(bs, device=DEVICE) % (gamma + 1)).to(cl_dtype)
        tc._parity(
            dspark_verify_window.BuildOutTokens,
            draft_tokens=_ri(0, VOCAB, (bs, gamma)),
            correct_len=cl,
            bonus=_ri(0, VOCAB, (bs,)),
            verify_num_draft_tokens=gamma + 1,
            gamma=gamma,
        )


def _case_build_ragged_verify_window(tc):
    torch.manual_seed(4)
    gamma, t, bs = 5, 6, 8
    verify_lens = _ri(1, t + 1, (bs,), torch.int32)
    batch = types.SimpleNamespace(
        seq_lens=_ri(1, 20, (bs,)),
        req_pool_indices=torch.randperm(bs + 3, device=DEVICE)[:bs],
    )
    model_runner = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=_ri(0, 1_000_000, (bs + 3, 64), torch.int32)
        )
    )
    for graph_num_tokens in (bs * t, (bs + 3) * t):  # tight and bucket padding
        tc._parity(
            dspark_verify_window.BuildRaggedVerifyWindow,
            batch=batch,
            layout=_layout(verify_lens, graph_num_tokens),
            draft_block_ids=_ri(0, VOCAB, (bs, gamma)),
            draft_tokens=_ri(0, VOCAB, (bs, gamma)),
            bs=bs,
            device=DEVICE,
            verify_num_draft_tokens=t,
            model_runner=model_runner,
        )


def _case_build_step_local(tc):
    torch.manual_seed(5)
    for org_width, per_partition, bias_dtype in (
        (32320, 32384, torch.bfloat16),
        (5000, 8192, torch.float32),
    ):
        bias = (torch.randn(3, org_width, device=DEVICE) * 3.0).to(bias_dtype)
        base = torch.randn(3, per_partition, device=DEVICE)
        got, _ = tc._parity(
            dspark_draft_model.BuildStepLocal, bias=bias, base_local=base
        )
        # Padding columns beyond org_width must stay pure base.
        tc.assertTrue(torch.equal(got[:, org_width:], base[:, org_width:]))


def _case_cap_correct_len(tc):
    torch.manual_seed(6)
    bs, nd = 64, 6
    verify_lens = _ri(1, nd + 1, (bs,), torch.int32)
    for cl_dtype in (torch.int32, torch.int64):
        cl = (torch.arange(bs, device=DEVICE) % (nd + 1)).to(cl_dtype)
        tc._parity(dspark_accept.CapCorrectLen, correct_len=cl, verify_lens=verify_lens)


def _case_causal_swa_page_indices(tc):
    swa, num_pool, pool_len, num_q = 128, 64, 600, 40
    g = torch.Generator(device=DEVICE).manual_seed(7)
    kw = dict(
        req_to_token=_ri(0, 40000, (num_pool, pool_len), torch.int32, g),
        full_to_swa_mapping=_ri(0, 1 << 20, (40000,), torch.int64, g),
        req_pool_indices_repeated=_ri(0, num_pool, (num_q,), torch.int32, g),
        swa_window=swa,
        page_index_aligned_size=96,
    )
    # Lens short of / straddling / beyond the SWA window boundary.
    for lo, hi in ((1, swa), (swa - 4, swa + 4), (swa + 1, pool_len)):
        lens = _ri(lo, hi, (num_q,), torch.int32, g)
        cls = attn_metadata_kernels.BuildCausalSwaPageIndices
        ref = cls.torch(seq_lens_casual=lens, **kw)
        got = cls.triton(seq_lens_casual=lens, **kw)
        tc.assertEqual(got.shape, ref.shape)
        tc.assertEqual(got.dtype, ref.dtype)
        # Parity holds on the attended region; padding slots must be -1.
        col = torch.arange(ref.shape[1], device=DEVICE).view(1, -1)
        attended = col < torch.clamp(lens, max=swa).view(-1, 1)
        tc.assertTrue(torch.equal(got[attended], ref[attended]))
        tc.assertTrue(bool((got[~attended] == -1).all()))


def _case_commit_inject_layout(tc):
    stride, num_pool, pool_len, n_full, bs = 7, 300, 400, 50000, 64
    g = torch.Generator(device=DEVICE).manual_seed(8)
    pool_perm = torch.randperm(num_pool, device=DEVICE, generator=g)
    kw = dict(
        req_pool_indices=pool_perm[:bs],
        req_to_token=_ri(0, n_full, (num_pool, pool_len), torch.int64, g),
        prefix_lens=_ri(1, pool_len - stride, (bs,), torch.int64, g),
        block_pos_offsets=torch.arange(stride, device=DEVICE),
        full_to_swa_mapping=_ri(0, 1 << 20, (n_full,), torch.int64, g),
        commit_lens=_ri(0, stride + 1, (bs,), torch.int32, g),
        stride=stride,
    )
    tc._parity(dspark_verify_window.BuildCommitInjectLayout, **kw)
    # commit_len edges: 0 masks the whole row to -1, stride keeps it all.
    kw.update(
        req_pool_indices=kw["req_pool_indices"][:2],
        prefix_lens=kw["prefix_lens"][:2],
        commit_lens=torch.tensor([0, stride], device=DEVICE, dtype=torch.int32),
    )
    edge = dspark_verify_window.BuildCommitInjectLayout.triton(**kw)
    swa_2d = edge.swa_loc.view(2, stride)
    tc.assertTrue(bool((swa_2d[0] == -1).all()))
    tc.assertTrue(bool((swa_2d[1] >= 0).all()))


def _case_commit_kv_proj(tc):
    hidden, head_dim, num_stages = 1024, 576, 3
    g = torch.Generator(device=DEVICE).manual_seed(9)
    linears = [
        _Bf16Linear(
            (torch.randn(head_dim, hidden, device=DEVICE, generator=g) * 0.02).to(
                torch.bfloat16
            )
        )
        for _ in range(num_stages)
    ]
    main_x = (torch.randn(56, hidden, device=DEVICE, generator=g) * 0.5).to(
        torch.bfloat16
    )
    cls = dspark_draft_model.CommitKvProj
    ref = cls.torch(main_x=main_x, wkv_linears=linears)
    got = cls.triton(main_x=main_x, wkv_linears=linears)
    tc.assertEqual(len(got), num_stages)
    for kv_got, kv_ref in zip(got, ref):
        tc.assertEqual(kv_got.shape, kv_ref.shape)
        tc.assertTrue(kv_got.is_contiguous())
        torch.testing.assert_close(kv_got.float(), kv_ref.float(), rtol=2e-2, atol=2e-3)
    # fp8 blockwise weight dequant path (2x3 grid of 128x128 blocks).
    out_dim, in_dim, block = 192, 384, 128
    w8 = torch.randn(out_dim, in_dim, device=DEVICE, generator=g).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(2, 3, device=DEVICE, generator=g) + 0.5
    sf = scale.repeat_interleave(block, 0)[:out_dim]
    sf = sf.repeat_interleave(block, 1)[:, :in_dim]
    expected = (w8.to(torch.float32) * sf).to(torch.bfloat16)
    stub = types.SimpleNamespace(weight=w8, weight_scale_inv=scale)
    tc._eq(dspark_draft_model._dequant_linear_weight(stub), expected)


def _case_compact_layout(tc):
    torch.manual_seed(10)
    gamma, t, bs = 5, 6, 64
    verify_lens = _ri(1, t + 1, (bs,), torch.int32)
    total = int(verify_lens.sum().item())
    for padded_total in (total, bs * t):  # exact and bucket padding
        tc._parity(
            dspark_verify_window.CompactRowIndex,
            verify_lens=verify_lens,
            padded_total=padded_total,
            device=DEVICE,
        )
        tc._parity(
            dspark_verify_window.CompactVerifyIds,
            draft_block_ids=_ri(0, VOCAB, (bs, gamma)),
            draft_tokens=_ri(0, VOCAB, (bs, gamma)),
            layout=_layout(verify_lens, padded_total),
            device=DEVICE,
        )


def _case_swa_page_indices(tc):
    torch.manual_seed(11)
    block_size, num_q, max_reqs, n_full = 5, 320, 300, 50000
    _, gather = tc._parity(
        dspark_attn_metadata.ComputeDsparkWindowGather,
        seq_lens_casual=_ri(1, 300, (num_q,), torch.int32),
        req_pool_indices_repeated=_ri(0, max_reqs, (num_q,)),
        block_size=block_size,
        swa_window=128,
    )
    tc._parity(
        dspark_attn_metadata.BuildDsparkSwaPageIndices,
        req_to_token=_ri(0, n_full, (max_reqs, 400), torch.int32),
        full_to_swa_mapping=_ri(0, 20000, (n_full,), torch.int32),
        req_pool_indices_per_request=gather.req_pool_indices_per_request,
        offsets=gather.offsets,
        invalid=gather.invalid,
        out_loc=_ri(0, n_full, (num_q,)),
        context_lens=gather.context_lens,
        block_size=block_size,
        swa_window=128,
        page_index_aligned_size=64,
    )


def _case_expand_prefill_causally(tc):
    torch.manual_seed(12)
    # Vectorized branch: ragged extends with padded token count.
    bs = 64
    extend = _ri(1, 8, (bs,))
    num_tokens = int(extend.sum())
    req_pool_indices = torch.randperm(512, device=DEVICE)[:bs]
    seq_lens = _ri(8, 500, (bs,))
    tc._parity(
        attn_metadata_kernels.ExpandPrefillCausally,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_seq_lens=extend,
        extend_start_loc=torch.cumsum(extend, dim=0) - extend,
        seq_lens_cpu=None,
        extend_seq_lens_cpu=None,
        num_tokens=num_tokens,
        padded_num_tokens=num_tokens + 5,
    )
    # Loop branch: uniform extend with CPU lens and no padding.
    bs2, block = 8, 6
    tc._parity(
        attn_metadata_kernels.ExpandPrefillCausally,
        req_pool_indices=req_pool_indices[:bs2],
        seq_lens=seq_lens[:bs2],
        extend_seq_lens=torch.full((bs2,), block, device=DEVICE),
        extend_start_loc=None,
        seq_lens_cpu=[int(x) for x in seq_lens[:bs2].tolist()],
        extend_seq_lens_cpu=[block] * bs2,
        num_tokens=bs2 * block,
        padded_num_tokens=None,
    )


def _case_finalize_accept_lens(tc):
    torch.manual_seed(13)
    bs = 64
    for prefix_dtype in (torch.int32, torch.int64):
        tc._parity(
            dspark_accept.FinalizeAcceptLens,
            correct_len=_ri(0, 7, (bs,), torch.int32),
            cap_trim_lens=_ri(0, 4, (bs,)),
            prefix_lens=_ri(1, 4000, (bs,), prefix_dtype),
        )


def _case_mixed_accept_select(tc):
    torch.manual_seed(14)
    bs = 64
    # Mixed dtypes between the greedy and sampling lanes.
    tc._parity(
        dspark_accept.SelectMixedAccept,
        greedy_mask=torch.rand(bs, device=DEVICE) < 0.5,
        greedy_len=_ri(0, 7, (bs,)),
        greedy_bonus=_ri(0, 100000, (bs,)),
        greedy_trim=_ri(0, 4, (bs,)),
        sampling_len=_ri(0, 7, (bs,), torch.int32),
        sampling_bonus=_ri(0, 100000, (bs,)),
        sampling_trim=_ri(0, 4, (bs,), torch.int32),
    )


def _case_padded_to_bucket(tc):
    torch.manual_seed(15)
    for bs, padded_bs, graph_num_tokens in ((3, 6, 16), (2, 8, 16), (8, 128, 768)):
        verify_lens = _ri(1, 7, (bs,), torch.int32)
        if int(verify_lens.sum()) > graph_num_tokens:
            verify_lens = torch.ones(bs, dtype=torch.int32, device=DEVICE)
        got, _ = tc._parity(
            ragged_verify_kernels.PaddedToBucket,
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
        )
        # Padding rows must absorb exactly the leftover budget.
        tc.assertEqual(int(got.to(torch.int64).sum()), graph_num_tokens)
        if padded_bs > bs:
            tc.assertTrue(torch.equal(got[:bs], verify_lens))


def _case_page_table_positions(tc):
    num_pool, pool_len = 128, 4096
    g = torch.Generator(device=DEVICE).manual_seed(16)
    req_to_token = _ri(0, 1 << 20, (num_pool, pool_len), torch.int32, g)
    # Large page + non-pool-aligned max_seq_len, then page_size 1.
    for num_q, page_size, max_seq_len in ((300, 64, 4000), (56, 1, 4096)):
        tc._parity(
            attn_metadata_kernels.BuildPageTablePositions,
            req_to_token=req_to_token,
            req_pool_indices_repeated=_ri(0, num_pool, (num_q,), torch.int32, g),
            seq_lens_casual=_ri(1, pool_len, (num_q,), torch.int64, g),
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=128,
        )


def _case_qo_indptr(tc):
    torch.manual_seed(17)
    cls = ragged_verify_kernels.BuildQoIndptr
    for dtype in (torch.int32, torch.int64):
        verify_lens = _ri(1, 8, (129,), dtype)  # straddles the 128 block
        ref = cls.torch(verify_lens=verify_lens)
        got = cls.triton(verify_lens=verify_lens.to(torch.int32))
        tc._eq(got, ref)
    # Aliasing regression: the two outputs must not share storage.
    vl = torch.tensor([3, 1, 5], device=DEVICE, dtype=torch.int32)
    got = cls.triton(verify_lens=vl)
    got.extend_start_loc.fill_(-7)
    tc.assertEqual(got.qo_indptr[:2].tolist(), [0, 3])


def _case_sample_step_tokens(tc):
    torch.manual_seed(18)
    cls = dspark_draft_model.SampleStepTokens
    # Injected noise makes stochastic sampling exactly comparable.
    for vocab, dtype in ((130000, torch.bfloat16), (5003, torch.float32)):
        bs = 3
        tc._parity(
            cls,
            step_logits=(torch.randn(bs, vocab, device=DEVICE) * 4.0).to(dtype),
            temperatures=torch.rand(bs, device=DEVICE) + 0.5,
            greedy_mask=(torch.arange(bs, device=DEVICE) % 2) == 0,
            exp_noise=torch.empty(bs, vocab, device=DEVICE).exponential_(1),
        )
    # Greedy tie straddling a triton block boundary picks the smaller index.
    logits = torch.zeros(1, 2050, device=DEVICE)
    logits[0, 1000] = logits[0, 1100] = 5.0
    tokens = cls.triton(
        step_logits=logits,
        temperatures=torch.tensor([1.0], device=DEVICE),
        greedy_mask=torch.tensor([True], device=DEVICE),
        exp_noise=torch.ones(1, 2050, device=DEVICE),
    )
    tc.assertEqual(tokens.item(), 1000)
    # Non-contiguous strided cropped view must match its contiguous copy.
    view = (torch.randn(2, 129536, device=DEVICE) * 4.0)[:, :VOCAB]
    tc.assertFalse(view.is_contiguous())
    kw = dict(
        temperatures=torch.rand(2, device=DEVICE) + 0.5,
        greedy_mask=torch.tensor([True, False], device=DEVICE),
        exp_noise=torch.empty(2, VOCAB, device=DEVICE).exponential_(1),
    )
    tc._eq(
        cls.triton(step_logits=view, **kw),
        cls.triton(step_logits=view.contiguous(), **kw),
    )


def _case_scatter_compact_to_strided(tc):
    torch.manual_seed(19)
    t, bs, dim = 6, 8, 4096
    verify_lens = _ri(1, t + 1, (bs,), torch.int32)
    total = int(verify_lens.sum().item())
    for graph_num_tokens in (total, bs * t):  # exact and bucket padding
        compact = torch.randn(
            graph_num_tokens, dim, dtype=torch.bfloat16, device=DEVICE
        )
        tc._parity(
            dspark_verify_window.ScatterCompactToStrided,
            compact=compact,
            layout=_layout(verify_lens, graph_num_tokens),
            fill_value=0.0,
            verify_num_draft_tokens=t,
        )


def _case_schedule_verify_lens_topk(tc):
    torch.manual_seed(20)
    gamma, bs = 5, 64
    cfg = DSparkScheduleConfig(gamma=gamma)
    cls = dspark_schedule.ScheduleVerifyLensTopk
    base = torch.rand(bs, gamma, device=DEVICE)
    confidences = (
        torch.full((bs, gamma), 0.5, device=DEVICE),  # all-ties
        (base * 4).floor() / 4,  # coarse quantization
        torch.where(base < 0.3, torch.zeros_like(base), base),  # invalid zeros
    )
    for confidence in confidences:
        for budget in (0, 1, 3, 7, 1000):
            tc._parity(cls, confidence=confidence, budget=budget, cfg=cfg)


def _case_softmax_temp(tc):
    g = torch.Generator(device=DEVICE).manual_seed(21)
    cls = dspark_accept.SoftmaxTemp
    # bf16 logits, non-power-of-two rows_per_request, full vocab.
    logits = (torch.randn(56, VOCAB, device=DEVICE, generator=g) * 8.0).to(
        torch.bfloat16
    )
    temps = (torch.rand(8, device=DEVICE, generator=g) * 1.5 + 0.05).float()
    ref = cls.torch(logits=logits, temperatures=temps, rows_per_request=7)
    got = cls.triton(logits=logits, temperatures=temps, rows_per_request=7)
    tc.assertEqual(got.dtype, torch.float32)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        got.sum(dim=-1), torch.ones_like(got.sum(dim=-1)), rtol=1e-5, atol=1e-5
    )
    # Column-shaped (bs, 1) temperatures.
    logits2 = torch.randn(6, 512, device=DEVICE, generator=g).to(torch.bfloat16)
    temps2 = (torch.rand(2, 1, device=DEVICE, generator=g) + 0.3).float()
    ref2 = cls.torch(logits=logits2, temperatures=temps2, rows_per_request=3)
    got2 = cls.triton(logits=logits2, temperatures=temps2, rows_per_request=3)
    torch.testing.assert_close(got2, ref2, rtol=1e-5, atol=1e-7)


_CASES = [
    ("accept_greedy", _case_accept_greedy),
    ("accept_sampling", _case_accept_sampling),
    ("build_block_seq_lens_causal", _case_build_block_seq_lens_causal),
    ("build_out_tokens", _case_build_out_tokens),
    ("build_ragged_verify_window", _case_build_ragged_verify_window),
    ("build_step_local", _case_build_step_local),
    ("cap_correct_len", _case_cap_correct_len),
    ("causal_swa_page_indices", _case_causal_swa_page_indices),
    ("commit_inject_layout", _case_commit_inject_layout),
    ("commit_kv_proj", _case_commit_kv_proj),
    ("compact_layout", _case_compact_layout),
    ("swa_page_indices", _case_swa_page_indices),
    ("expand_prefill_causally", _case_expand_prefill_causally),
    ("finalize_accept_lens", _case_finalize_accept_lens),
    ("mixed_accept_select", _case_mixed_accept_select),
    ("padded_to_bucket", _case_padded_to_bucket),
    ("page_table_positions", _case_page_table_positions),
    ("qo_indptr", _case_qo_indptr),
    ("sample_step_tokens", _case_sample_step_tokens),
    ("scatter_compact_to_strided", _case_scatter_compact_to_strided),
    ("schedule_verify_lens_topk", _case_schedule_verify_lens_topk),
    ("softmax_temp", _case_softmax_temp),
]


class TestDsparkKernelParity(CustomTestCase):
    def _eq(self, got, ref):
        """Exact comparison of tensors, tuples, and msgspec result structs."""
        if isinstance(ref, tuple):
            for g, r in zip(got, ref):
                self._eq(g, r)
        elif hasattr(ref, "__struct_fields__"):
            for name in ref.__struct_fields__:
                self._eq(getattr(got, name), getattr(ref, name))
        elif isinstance(ref, torch.Tensor):
            self.assertEqual(got.dtype, ref.dtype)
            self.assertTrue(torch.equal(got, ref))
        else:
            self.assertEqual(got, ref)

    def _parity(self, cls, **kw):
        got, ref = cls.triton(**kw), cls.torch(**kw)
        self._eq(got, ref)
        return got, ref

    def test_all_kernels_triton_matches_torch(self):
        for name, case in _CASES:
            with self.subTest(kernel=name):
                case(self)


def _candidate_reference(base, ids, prev, embeds, weight, table):
    """Independent subset oracle; no full-vocabulary equivalence is assumed."""
    candidates = torch.cat((ids.long(), table[prev.long()].long()), dim=1)
    selected = weight[candidates].float()
    bias = torch.zeros_like(candidates, dtype=torch.float32)
    for start in range(0, weight.shape[1], 32):
        bias += (
            selected[:, :, start : start + 32]
            * embeds[:, None, start : start + 32].float()
        ).sum(-1)
    scores = base.gather(1, candidates).float() + bias
    nan = scores.isnan()
    winners = torch.where(
        nan.any(-1, keepdim=True), nan, scores == scores.amax(-1, keepdim=True)
    )
    return candidates.masked_fill(~winners, base.shape[1]).amin(-1)


class TestMarkovCandidates(CustomTestCase):
    def _inputs(self, rank=33, dtype=torch.bfloat16, k=7, m=5, batch=4):
        torch.manual_seed(22)
        vocab = 257
        # Strided rows and previous tokens exercise block slices used by serving.
        base = torch.randn(batch * 2, vocab, device=DEVICE, dtype=dtype)[::2]
        ids = _ri(0, vocab, (batch * 2, k))[::2]
        prev = _ri(0, vocab, (batch * 2,))[::2]
        embeds = torch.randn(batch * 2, rank, device=DEVICE, dtype=dtype)[::2]
        weight = torch.randn(vocab * 2, rank, device=DEVICE, dtype=dtype)[::2]
        table = _ri(0, vocab, (vocab * 2, m), dtype=torch.int32)[::2]
        return base, ids, prev, embeds, weight, table

    def test_subset_rank_tails_and_dtypes(self):
        for rank, dtype, k, m in (
            (1, torch.float32, 1, 1),
            (31, torch.float16, 7, 5),
            (33, torch.bfloat16, 7, 5),
            (256, torch.bfloat16, 128, 128),
        ):
            with self.subTest(rank=rank, dtype=dtype, k=k, m=m):
                args = self._inputs(rank, dtype, k, m)
                torch.testing.assert_close(
                    markov_candidate_step(*args),
                    _candidate_reference(*args),
                    rtol=0,
                    atol=0,
                )

    def test_ties_duplicates_padding_and_nan(self):
        args = self._inputs()
        base, ids, prev, embeds, weight, table = args
        weight.zero_()
        ids.fill_(9)
        table.fill_(3)
        for value in (0.0, float("-inf")):
            base.fill_(value)
            # K+M=12 leaves padded lanes: fake ID zero must never win.
            self.assertTrue(
                torch.equal(markov_candidate_step(*args), prev.new_full((4,), 3))
            )
        base.fill_(0)
        base[:, 9] = float("nan")
        torch.testing.assert_close(
            markov_candidate_step(*args), _candidate_reference(*args)
        )

    def test_metadata_and_empty_batch(self):
        args = self._inputs()
        for index, bad in (
            (0, args[0].double()),
            (1, args[1].float()),
            (1, _ri(0, 257, (4, 129))),
            (0, args[0].unsqueeze(0)),
            (2, args[2][:2]),
            (3, args[3][:, ::2]),
            (4, args[4].T.contiguous().T),
            (5, args[5][:, :0]),
            (5, _ri(0, 257, (257, 129))),
            (0, args[0].cpu()),
        ):
            with self.subTest(index=index):
                invalid = list(args)
                invalid[index] = bad
                with self.assertRaises(ValueError):
                    markov_candidate_step(*invalid)
        empty = [x[:0] if i < 4 else x for i, x in enumerate(args)]
        result = markov_candidate_step(*empty)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, torch.int64)

    def test_graph_replay_reads_changed_inputs(self):
        args = self._inputs()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            markov_candidate_step(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = markov_candidate_step(*args)
        for anchor in (1, 23):
            args[2].fill_(anchor)
            args[0].normal_()
            args[3].normal_()
            graph.replay()
            torch.testing.assert_close(
                output, _candidate_reference(*args), rtol=0, atol=0
            )


class TestVanillaMarkovCandidates(CustomTestCase):
    @torch.no_grad()
    def test_live_table_and_block_graph_replay(self):
        from sglang.srt.models.dspark import VanillaMarkov

        torch.manual_seed(23)
        model = VanillaMarkov(vocab_size=257, markov_rank=33).to(
            device=DEVICE, dtype=torch.bfloat16
        )
        base = torch.randn(4, 3, 257, device=DEVICE, dtype=torch.bfloat16)
        anchors = _ri(0, 257, (4,))
        # Disabled by default: compare with the original full-vocab fused steps.
        self.assertEqual(model.candidate_k, 0)
        prev, expected = anchors, []
        for step in range(3):
            prev = dspark_draft_model.MarkovGreedyStep.execute(
                base_logits=base[:, step],
                prev_embeds=model.get_prev_embeddings(prev),
                w2_weight=model.markov_w2.weight,
            )
            expected.append(prev)
        torch.testing.assert_close(
            model.sample_block_greedy_fused(base, first_prev_tokens=anchors),
            torch.stack(expected, 1),
            rtol=0,
            atol=0,
        )
        model.prepare_candidates(7, 5)
        pointer = model.bias_top_ids.data_ptr()
        self.assertNotIn("bias_top_ids", model.state_dict())

        def check_table():
            # BF16 inputs converted to FP32 give an independent score reference.
            scores = model.markov_w1.weight.float() @ model.markov_w2.weight.float().T
            chosen = scores.gather(1, model.bias_top_ids.long())
            torch.testing.assert_close(
                chosen.sort(-1).values,
                scores.topk(5, dim=-1).values.sort(-1).values,
                rtol=0,
                atol=0,
            )

        def reference():
            ids = base.topk(7, dim=-1, sorted=False).indices
            prev, outputs = anchors, []
            for step in range(base.shape[1]):
                prev = _candidate_reference(
                    base[:, step],
                    ids[:, step],
                    prev,
                    model.get_prev_embeddings(prev),
                    model.markov_w2.weight,
                    model.bias_top_ids,
                )
                outputs.append(prev)
            return torch.stack(outputs, 1)

        check_table()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            model.sample_block_greedy_fused(base, first_prev_tokens=anchors)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model.sample_block_greedy_fused(base, first_prev_tokens=anchors)
        for anchor in (3, 17):
            anchors.fill_(anchor)
            graph.replay()
            torch.testing.assert_close(output, reference(), rtol=0, atol=0)
        old_table = model.bias_top_ids.clone()
        model.markov_w2.weight.neg_()
        model.prepare_candidates(7, 5)
        self.assertEqual(model.bias_top_ids.data_ptr(), pointer)
        self.assertFalse(torch.equal(old_table, model.bias_top_ids))
        check_table()
        graph.replay()
        torch.testing.assert_close(output, reference(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
