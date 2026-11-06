"""Correctness tests for grouped-head target-verify attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
from sglang.kernels.ops.attention.verify_mla_draft import (
    verify_shared_kv_fwd as verify_draft_fwd,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    _should_use_verify_shared_kv,
)
from sglang.srt.layers.attention.triton_draft_hip import init_draft_backend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_hip_version, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")

BF16_ATOL = 2e-2
BF16_RTOL = 1e-2
FP8_ATOL = 8e-2
FP8_RTOL = 2e-2


def _build_inputs(
    prefix_lens,
    l_ext,
    h_q,
    head_dim,
    v_head_dim,
    h_kv=1,
    cache_dtype=torch.bfloat16,
    latent_values=False,
    shuffle_cache=False,
    seed=0,
):
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    total_prefix = sum(prefix_lens)
    batch_size = len(prefix_lens)
    num_extend_tokens = batch_size * l_ext

    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

    q = randn(num_extend_tokens, h_q, head_dim)
    k = randn(num_extend_tokens, h_kv, head_dim)
    v = (
        k[..., :v_head_dim].contiguous()
        if latent_values
        else randn(num_extend_tokens, h_kv, v_head_dim)
    )
    cache_rows = total_prefix + 64 if shuffle_cache else total_prefix
    k_buffer = randn(cache_rows, h_kv, head_dim).to(cache_dtype)
    v_buffer = (
        k_buffer[..., :v_head_dim]
        if latent_values
        else randn(cache_rows, h_kv, v_head_dim).to(cache_dtype)
    )
    qo_indptr = torch.arange(
        0, num_extend_tokens + 1, l_ext, dtype=torch.int32, device=device
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(prefix_lens_t, dim=0)
    kv_indices = (
        torch.randperm(cache_rows, device=device, generator=generator)[:total_prefix]
        if shuffle_cache
        else torch.arange(total_prefix, dtype=torch.int64, device=device)
    )
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestVerifySharedKV(CustomTestCase):
    def _run_parity(
        self,
        head_dim,
        v_head_dim,
        h_q=4,
        h_kv=1,
        is_causal=True,
        cache_dtype=torch.bfloat16,
        k_scale=1.0,
        v_scale=1.0,
        l_ext=4,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        prefix_lens=(512, 2048),
        sliding_window_size=-1,
    ):
        inputs = _build_inputs(
            prefix_lens=list(prefix_lens),
            l_ext=l_ext,
            h_q=h_q,
            h_kv=h_kv,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            cache_dtype=cache_dtype,
        )
        case = dict(
            h_q=h_q,
            h_kv=h_kv,
            d=head_dim,
            dv=v_head_dim,
            l_ext=l_ext,
            causal=is_causal,
            mla=False,
            k_scale=k_scale,
            v_scale=v_scale,
            window=sliding_window_size,
        )
        reference = _extend(case, inputs, bf16_dot=False)
        actual = _extend(case, inputs, bf16_dot=False, kernel=verify_shared_kv_fwd)
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)

    def test_qwen3_5_tp_shapes(self):
        # Qwen3.5 has 32 global query heads. TP8, TP4, and InferenceX's TP2
        # configurations expose 4, 8, and 16 local query heads respectively,
        # all sharing one TP-local KV head.
        for h_q in (4, 8, 16):
            with self.subTest(h_q=h_q):
                self._run_parity(head_dim=256, v_head_dim=256, h_q=h_q)

    def test_qwen3_5_short_verify_widths(self):
        for l_ext in (1, 2, 3):
            with self.subTest(l_ext=l_ext):
                self._run_parity(head_dim=256, v_head_dim=256, l_ext=l_ext)

    @unittest.skipIf(
        get_hip_version()[:2] == (7, 0),
        "Triton 3.4 on ROCm 7.0 aborts gfx950 fp8 KV tl.dot "
        "(triton-lang/triton#8278). Remove once the image uses Triton >= 3.6.",
    )
    def test_qwen3_5_fp8_kv_cache(self):
        self._run_parity(
            head_dim=256,
            v_head_dim=256,
            h_q=8,
            cache_dtype=torch.float8_e4m3fn,
            k_scale=0.5,
            v_scale=0.25,
            atol=FP8_ATOL,
            rtol=FP8_RTOL,
        )

    def test_kimi_k3_absorbed_mla_shape(self):
        self._run_parity(head_dim=576, v_head_dim=512)

    def test_kimi_k3_dspark_gqa_shape(self):
        self._run_parity(
            head_dim=64,
            v_head_dim=64,
            h_q=8,
            h_kv=2,
            is_causal=False,
            l_ext=7,
        )

    def test_windowed_verify_matches_the_extend_kernel(self):
        """Windowed verify must match the extend kernel at the M3.1 draft's window;
        prefix lists are pre-windowed by the backend, at most one window long."""
        for prefix_lens, window in (
            ((300, 4095), 4095),
            # The short window hides 0-6 of 16 keys per query, so an off-by-one shows.
            ((16, 13), 16),
            # Past 16 requests the combine merges several draft rows per program.
            ((300, 4095) * 9, 4095),
        ):
            with self.subTest(window=window):
                self._run_parity(
                    head_dim=128,
                    v_head_dim=128,
                    h_q=16,
                    is_causal=False,
                    l_ext=7,
                    prefix_lens=prefix_lens,
                    sliding_window_size=window,
                )

    def test_strided_draft_rows_match_contiguous_rows(self):
        """K/V rows split out of a packed qkv projection (the attention hands them
        over uncopied) give the bits contiguous rows give."""
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = _build_inputs(
            prefix_lens=[300, 4095], l_ext=7, h_q=16, head_dim=128, v_head_dim=128
        )
        tokens = q.shape[0]
        qkv = torch.cat([q.reshape(tokens, -1), k[:, 0], v[:, 0]], dim=-1)
        k_rows = qkv[:, 16 * 128 : 17 * 128].view(tokens, 1, 128)
        v_rows = qkv[:, 17 * 128 :].view(tokens, 1, 128)
        self.assertFalse(v_rows.is_contiguous())
        outs = []
        for k_ext, v_ext in (
            (k_rows, v_rows),
            (k_rows.contiguous(), v_rows.contiguous()),
        ):
            out = torch.empty_like(q)
            self.assertTrue(
                verify_shared_kv_fwd(
                    q,
                    k_ext,
                    v_ext,
                    out,
                    k_buffer,
                    v_buffer,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    None,
                    False,
                    None,
                    7,
                    1.0,
                    1.0,
                    sm_scale=128**-0.5,
                    sliding_window_size=4095,
                    max_bs=2,
                    allow_sliding_window=True,
                )
            )
            outs.append(out)
        self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_sliding_window_is_declined_unless_the_caller_opts_in(self):
        """Only the M3.1 draft opts in; every other windowed caller keeps the extend kernel."""
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = _build_inputs(
            prefix_lens=[16], l_ext=7, h_q=16, head_dim=128, v_head_dim=128
        )
        self.assertFalse(
            verify_shared_kv_fwd(
                q,
                k,
                v,
                torch.empty_like(q),
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                False,
                None,
                7,
                1.0,
                1.0,
                sliding_window_size=16,
            )
        )

    def test_rejects_non_power_of_two_kv_group(self):
        inputs = _build_inputs(
            prefix_lens=[512],
            l_ext=4,
            h_q=6,
            h_kv=2,
            head_dim=256,
            v_head_dim=256,
        )
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = inputs
        output = torch.empty_like(q)
        self.assertFalse(
            verify_shared_kv_fwd(
                q,
                k,
                v,
                output,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                True,
                None,
                4,
                1.0,
                1.0,
            )
        )

    @patch(
        "sglang.srt.layers.attention.triton_backend.is_gfx95_supported",
        return_value=True,
    )
    @patch("sglang.srt.layers.attention.triton_backend.get_parallel")
    def test_backend_dispatch_gate(self, get_parallel_mock, _is_gfx95_mock):
        get_parallel_mock.return_value = SimpleNamespace(
            attn_tp_size=8, attn_dcp_size=1
        )

        def model_config(architecture, local_kv_heads=1):
            return SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture]),
                get_num_kv_heads=lambda _tp, _dcp: local_kv_heads,
            )

        qwen = model_config("Qwen3_5MoeForCausalLM")
        self.assertTrue(_should_use_verify_shared_kv(qwen, 1, False, True))
        self.assertFalse(_should_use_verify_shared_kv(qwen, 2, False, True))
        self.assertFalse(_should_use_verify_shared_kv(qwen, 1, False, False))
        self.assertFalse(
            _should_use_verify_shared_kv(
                model_config("Qwen3_5MoeForCausalLM", local_kv_heads=2),
                1,
                False,
                True,
            )
        )
        self.assertFalse(
            _should_use_verify_shared_kv(
                model_config("LlamaForCausalLM"), 1, False, True
            )
        )
        self.assertTrue(
            _should_use_verify_shared_kv(
                model_config("KimiK3ForConditionalGeneration"), 1, True, False
            )
        )
        with patch(
            "sglang.srt.layers.attention.triton_backend.is_gfx95_supported",
            return_value=False,
        ):
            self.assertFalse(_should_use_verify_shared_kv(qwen, 1, False, True))


# FP8 q/probability rounding is materially larger than BF16 dot rounding.
BF16_ERR = 1e-2
# Triton 3.4 on ROCm 7.0 aborts gfx950 FP8 dots (triton-lang/triton#8278).
FP8_DOT_OK = get_hip_version()[:2] != (7, 0)

MLA = dict(h_q=12, h_kv=1, d=576, dv=512, l_ext=8, causal=True, mla=True)

EXTEND_CASES = {
    "mla5_window": dict(MLA, prefix_lens=(3000, 1500, 5), window=1023),
    "mla_full": dict(MLA, prefix_lens=(3000, 300, 5)),
}


def _inputs(case, cache_dtype, seed=0):
    return _build_inputs(
        prefix_lens=case["prefix_lens"],
        l_ext=case["l_ext"],
        h_q=case["h_q"],
        head_dim=case["d"],
        v_head_dim=case["dv"],
        h_kv=case["h_kv"],
        cache_dtype=cache_dtype,
        latent_values=case["mla"],
        shuffle_cache=True,
        seed=seed,
    )


def _scale(case):
    # MLA scales by the 128 + 64 qk head of the target, not the latent width.
    return (128 + 64) ** -0.5 if case["mla"] else case["d"] ** -0.5


def _reference(case, q, k, v, k_buffer, v_buffer, kv_indptr, kv_indices, scale):
    """fp32 attention of each draft row over its request's prefix and block, the
    cache read exactly (FP8 and BF16 values are exact in fp32) and descaled.
    The block is causal or not; a window keeps the keys of [p - window, p]."""
    h_q, h_kv, l_ext = case["h_q"], case["h_kv"], case["l_ext"]
    window = case.get("window", -1)
    k_scale, v_scale = case.get("k_scale", 1.0), case.get("v_scale", 1.0)
    bounds = kv_indptr.tolist()
    out = []
    for b in range(len(bounds) - 1):
        locs = kv_indices[bounds[b] : bounds[b + 1]]
        prefix = locs.numel()
        rows = slice(b * l_ext, (b + 1) * l_ext)
        keys = torch.cat([k_buffer[locs].float() * k_scale, k[rows].float()])
        vals = torch.cat([v_buffer[locs].float() * v_scale, v[rows].float()])
        q_rows = q[rows].float().unflatten(1, (h_kv, h_q // h_kv))
        s = torch.einsum("lkgd,tkd->kglt", q_rows, keys) * scale
        q_pos = prefix + torch.arange(l_ext, device="cuda")
        k_pos = torch.arange(prefix + l_ext, device="cuda")
        seen = torch.ones(l_ext, prefix + l_ext, dtype=torch.bool, device="cuda")
        if case["causal"]:
            seen &= k_pos[None, :] <= q_pos[:, None]
        if window > 0:
            seen &= k_pos[None, :] >= q_pos[:, None] - window
        p = torch.softmax(s.masked_fill(~seen, float("-inf")), dim=-1)
        out.append(torch.einsum("kglt,tkd->lkgd", p, vals).flatten(1, 2))
    return torch.cat(out)


def _extend(case, inputs, bf16_dot, kernel=extend_attention_fwd):
    q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = inputs
    out = torch.empty(*q.shape[:2], case["dv"], dtype=q.dtype, device="cuda")
    ran = kernel(
        q,
        k,
        v,
        out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        case["causal"],
        None,
        case["l_ext"],
        case.get("k_scale", 1.0),
        case.get("v_scale", 1.0),
        sm_scale=_scale(case),
        sliding_window_size=case.get("window", -1),
        **({"bf16_dot": bf16_dot} if kernel is not verify_shared_kv_fwd else {}),
        **(
            {"max_bs": len(kv_indptr) - 1} if kernel is not extend_attention_fwd else {}
        ),
        **(
            {"allow_sliding_window": case.get("window", -1) > 0}
            if kernel is verify_shared_kv_fwd
            else {}
        ),
    )
    if kernel is not extend_attention_fwd:
        assert ran is True
    return out


def _err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _backend_extend(case, inputs, bf16_dot):
    # Exercise the real dispatcher as well as the kernel: losing bf16_dot must
    # fail the FP32 accuracy check, not merely a recorded-kwargs assertion.
    q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = inputs
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.draft_bf16_dot = bf16_dot
    backend.use_dense_fp8_chunked_prefill = False
    backend.use_verify_shared_kv = False
    backend.use_verify_splitkv = False
    backend.verify_mla_draft_fwd = None
    backend.extend_attention_fwd = extend_attention_fwd
    backend.dcp_size = 1
    backend.enable_deterministic = False
    backend.dllm_attention = None
    backend.page_size = 1
    read = SimpleNamespace(
        kv=lambda: (k_buffer, v_buffer), key=lambda: k_buffer, value=lambda: v_buffer
    )
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: k_buffer,
        get_value_buffer=lambda _: v_buffer,
        layer_router=SimpleNamespace(
            read=lambda _: read,
            route=lambda _: SimpleNamespace(sub_pool=None),
            sub_pool=lambda _: None,
        ),
    )
    backend.forward_metadata = SimpleNamespace(
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        window_kv_indptr=kv_indptr,
        window_kv_indices=kv_indices,
        window_kv_offsets=None,
        qo_indptr=qo_indptr,
        custom_mask=None,
        mask_indptr=None,
        max_extend_len=case["l_ext"],
    )
    layer = SimpleNamespace(
        layer_id=0,
        qk_head_dim=case["d"],
        v_head_dim=case["dv"],
        tp_q_head_num=case["h_q"],
        k_scale=None,
        v_scale=None,
        logit_capping_method="tanh",
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=case.get("window", -1),
        scaling=_scale(case),
        xai_temperature_len=-1,
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        mha_one_shot=False,
        extend_seq_lens_cpu=None,
    )
    return backend.forward_extend(
        q, k, v, layer, forward_batch, save_kv_cache=False
    ).view(*q.shape[:2], case["dv"])


@unittest.skipUnless(is_hip(), "ROCm DFLASH draft kernels")
class TestDraftAttention(CustomTestCase):
    def test_fp8_cache_bf16_dot_matches_fp32(self):
        for name in EXTEND_CASES:
            with self.subTest(case=name):
                case = EXTEND_CASES[name]
                inputs = _inputs(case, fp8_dtype)
                q, k, v, k_buffer, v_buffer, _, kv_indptr, kv_indices = inputs
                ref = _reference(
                    case,
                    q,
                    k,
                    v,
                    k_buffer,
                    v_buffer,
                    kv_indptr,
                    kv_indices,
                    _scale(case),
                )

                out = _backend_extend(case, inputs, bf16_dot=True)
                err = _err(out, ref)
                assert err < BF16_ERR, err
                torch.testing.assert_close(
                    out.float(), ref, atol=BF16_ATOL, rtol=BF16_RTOL
                )

                if FP8_DOT_OK:
                    # The default still runs FP8 q and probabilities over an FP8 cache.
                    default = _backend_extend(case, inputs, bf16_dot=False)
                    default_err = _err(default, ref)
                    assert not torch.equal(default, out)
                    assert default_err > 2 * err, (default_err, err)

    def test_bf16_cache_unchanged_by_bf16_dot(self):
        for name in EXTEND_CASES:
            with self.subTest(case=name):
                case = EXTEND_CASES[name]
                inputs = _inputs(case, torch.bfloat16, seed=1)
                torch.testing.assert_close(
                    _extend(case, inputs, bf16_dot=True),
                    _extend(case, inputs, bf16_dot=False),
                    rtol=0,
                    atol=0,
                )

    def test_draft_backend_gate(self):
        for algorithm, expected in (
            (SpeculativeAlgorithm.DFLASH, True),
            (SpeculativeAlgorithm.DSPARK, True),
            (SpeculativeAlgorithm.EAGLE, False),
        ):
            with self.subTest(algorithm=algorithm):
                backend = SimpleNamespace(use_mla=False, dcp_size=1)
                init_draft_backend(backend, SimpleNamespace(spec_algorithm=algorithm))
                assert backend.draft_bf16_dot is expected

    def test_verify_matches_extend(self):
        cases = [
            ((65536, 3000), 8, -1, False),
            ((600, 1024, 1100), 6, 1024, False),
            ((600, 1024, 1100), 6, 1024, True),
            ((600, 1024, 1100), 8, 1024, False),
            ((600, 1024, 1100), 8, 1024, True),
        ]
        for prefix_lens, l_ext, window, bf16_dot in cases:
            with self.subTest(l_ext=l_ext, window=window, bf16_dot=bf16_dot):
                case = dict(
                    MLA, mla=False, prefix_lens=prefix_lens, l_ext=l_ext, window=window
                )
                inputs = _inputs(case, torch.float8_e4m3fn)
                actual = _extend(case, inputs, bf16_dot, kernel=verify_draft_fwd)
                expected = _extend(case, inputs, bf16_dot)
                torch.testing.assert_close(actual, expected, atol=8e-2, rtol=2e-2)

    def test_verify_rejects_window_narrower_than_block(self):
        case = dict(MLA, mla=False, prefix_lens=(256,))
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = _inputs(
            case, torch.bfloat16
        )
        out = torch.empty(*q.shape[:2], case["dv"], dtype=q.dtype, device=q.device)
        assert not verify_draft_fwd(
            q, k, v, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
            None, True, None, 8, 1.0, 1.0, sliding_window_size=4,
        )  # fmt: skip


if __name__ == "__main__":
    unittest.main()
