import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.flashinfer_prefill_kv_dequant import (
    FlashInferPrefillKVDequant,
)
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and is_flashinfer_available(),
    "CUDA and FlashInfer required",
)
class TestFlashInferPrefillKVDequant(unittest.TestCase):
    def test_paged_attention_matches_bf16_pool(self):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        torch.manual_seed(42)
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        factory = lambda: BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            for page_size in [1, 16, 256]:
                with self.subTest(dtype=dtype, page_size=page_size):
                    kv_lens, q_lens = [1041, 525, 769], [33, 45, 17]
                    page_counts = [(n + page_size - 1) // page_size for n in kv_lens]
                    nnz = sum(page_counts)
                    # K/V are interleaved: pages are contiguous inside, but
                    # page strides are twice the packed row size.
                    pool = torch.randn(
                        nnz + 11, 2, page_size, 1, 256, device="cuda"
                    ).to(dtype)
                    k, v = pool[:, 0], pool[:, 1]
                    pages = torch.randperm(nnz + 11, device="cuda")[:nnz].to(
                        torch.int32
                    )
                    pages[-1] = pages[0]  # Shared physical page must stay duplicated.
                    indices = torch.cat(
                        [
                            pages,
                            torch.full((23,), 999999, dtype=torch.int32, device="cuda"),
                        ]
                    )
                    qo = torch.tensor([0, 33, 78, 95], dtype=torch.int32, device="cuda")
                    indptr = torch.tensor(
                        [0, page_counts[0], sum(page_counts[:2]), nnz],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    last = torch.tensor(
                        [(n - 1) % page_size + 1 for n in kv_lens],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    kwargs = dict(
                        qo_indptr=qo,
                        paged_kv_indptr=indptr,
                        paged_kv_indices=indices,
                        paged_kv_last_page_len=last,
                        num_qo_heads=6,
                        num_kv_heads=1,
                        head_dim_qk=256,
                        page_size=page_size,
                        q_data_type=torch.bfloat16,
                        kv_data_type=dtype,
                    )
                    original = factory()
                    original.begin_forward(**kwargs)
                    helper = FlashInferPrefillKVDequant(factory, 1, 0, 16384)
                    plan = helper.prepare(
                        num_new_tokens=sum(q_lens),
                        num_pages=nnz,
                        use_ragged=False,
                        spec_info=None,
                        original_wrapper=original,
                        plan_kwargs=kwargs,
                    )
                    self.assertIsNotNone(plan)
                    kb, vb = plan.gather((k, v))
                    self.assertTrue(
                        torch.equal(
                            kb, k.to(torch.bfloat16).index_select(0, pages.long())
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            vb, v.to(torch.bfloat16).index_select(0, pages.long())
                        )
                    )
                    q = torch.randn(
                        sum(q_lens), 6, 256, dtype=torch.bfloat16, device="cuda"
                    )
                    run_kwargs = dict(
                        causal=True, sm_scale=0.0625, k_scale=0.75, v_scale=1.5
                    )
                    out = plan.wrapper.forward(q, (kb, vb), **run_kwargs)
                    reference = factory()
                    reference.begin_forward(**dict(kwargs, kv_data_type=torch.bfloat16))
                    expected = reference.forward(
                        q, (k.to(torch.bfloat16), v.to(torch.bfloat16)), **run_kwargs
                    )
                    self.assertTrue(torch.equal(out, expected))
                    out_lse, lse = plan.wrapper.forward_return_lse(
                        q, (kb, vb), **run_kwargs
                    )
                    expected_lse, ref_lse = reference.forward_return_lse(
                        q, (k.to(torch.bfloat16), v.to(torch.bfloat16)), **run_kwargs
                    )
                    self.assertTrue(torch.equal(out_lse, expected_lse))
                    self.assertTrue(torch.equal(lse, ref_lse))
                    self.assertIsNone(
                        helper.prepare(
                            num_new_tokens=0,
                            num_pages=nnz,
                            use_ragged=False,
                            spec_info=None,
                            original_wrapper=original,
                            plan_kwargs=kwargs,
                        )
                    )
                    raw = original.forward(q, (k, v), **run_kwargs)
                    self.assertTrue(
                        torch.equal(raw, original.forward(q, (k, v), **run_kwargs))
                    )

    def test_backend_routes_enabled_and_following_fallback_batches(self):
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.test.kits.attention_unittest.attention_methods import (
            dense_attention as dense,
        )

        pool_class = dense.MHATokenToKVPool
        runner_init = dense.MockModelRunner.__init__

        def init_fp8(runner, *args, **kwargs):
            with patch.object(
                dense,
                "MHATokenToKVPool",
                side_effect=lambda **kw: pool_class(
                    **dict(kw, dtype=torch.float8_e4m3fn)
                ),
            ):
                runner_init(runner, *args, **kwargs)
            runner.kv_cache_dtype = torch.float8_e4m3fn
            runner.kv_cache_dtype_str = "fp8_e4m3"

        case = dense.DenseAttentionCase(
            name="fp8_prefill_dequant_routing",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=6,
            num_kv_heads=1,
            page_size=256,
            prefix_lens=(1024, 256),
            extend_lens=(32, 16),
        )
        with (
            envs.SGLANG_PREFILL_KV_DEQUANT.override(True),
            envs.SGLANG_PREFILL_KV_DEQUANT_MIN_TOKENS.override(1),
            envs.SGLANG_PREFILL_KV_DEQUANT_MIN_KV_TOKENS.override(0),
            envs.SGLANG_FLASHINFER_USE_PAGED.override(True),
            patch.object(dense.MockModelRunner, "__init__", init_fp8),
        ):
            fixture = dense.build_dense_attention_fixture(
                self,
                case,
                head_dim=256,
                hidden_size=64,
                max_context_len=2048,
                dtype=torch.bfloat16,
            )
            backend = fixture.backend
            original = backend.prefill_wrappers_paged[0]
            with patch.object(
                original,
                "forward",
                side_effect=AssertionError("eligible batch used FP8 wrapper"),
            ):
                out = dense.run_dense_fixture_eager(fixture)
            self.assertTrue(torch.isfinite(out).all())
            self.assertIsNotNone(backend.forward_metadata.kv_dequant_plan)
            self.assertEqual(
                backend.forward_metadata.kv_dequant_plan.pages.numel(),
                sum(case.seq_lens),
            )
            self.assertEqual(
                fixture.runner.token_to_kv_pool.get_kv_buffer(0)[0].dtype,
                torch.float8_e4m3fn,
            )
            backend.fp8_prefill_kv_dequant.min_tokens = 9999
            with patch.object(original, "forward", wraps=original.forward) as forward:
                dense.run_dense_fixture_eager(fixture)
            self.assertIsNone(backend.forward_metadata.kv_dequant_plan)
            forward.assert_called_once()
            self.assertEqual(forward.call_args.args[1][0].dtype, torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
