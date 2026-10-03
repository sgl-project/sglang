import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import flashinfer_backend as backend_module
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFlashInferLogitsSoftCap(CustomTestCase):
    def setUp(self):
        self.dtype = torch.float16
        self.heads, self.dim, self.length = 4, 64, 3
        self.workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.q = torch.zeros(3, self.heads, self.dim, dtype=self.dtype, device="cuda")
        self.q[..., 0] = 1
        self.k = torch.zeros_like(self.q)
        self.k[..., 0] = self.tensor([0, 2, 10], self.dtype)[:, None]
        self.v = (
            self.tensor([0, 1, 2], self.dtype)[:, None, None]
            .expand_as(self.q)
            .contiguous()
        )
        self.ptr = self.tensor([0, 3])
        self.indices = self.tensor([0, 1, 2])
        self.last_page = self.tensor([1])

    def tensor(self, values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device="cuda")

    def make_backend(self, cap):
        backend = backend_module.FlashInferAttnBackend.__new__(
            backend_module.FlashInferAttnBackend
        )
        backend.flashinfer_kv_cache_dtype = self.dtype
        backend.prefill_uses_dequant_workspace = False
        backend.decode_uses_dequant_workspace = False
        backend.is_dllm_model = False
        backend.num_wrappers = 1
        backend.dispatch_reason = None
        backend.kv_indptr = [self.tensor([0, 0])]
        backend.qo_indptr = [self.tensor([0, 0])]
        backend.kv_last_page_len = self.last_page
        backend._swa_kv_pool = None
        backend.dq_paged_kernel_lens = None
        backend.kv_index_translator = SimpleNamespace(reads_are_translated=False)
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda _: (self.k[:, None], self.v[:, None])
        )
        backend.prefill_wrapper_ragged = (
            backend_module.BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace, "NHD", backend="auto"
            )
        )
        config = SimpleNamespace(
            head_dim=self.dim,
            hf_text_config=SimpleNamespace(attn_logit_softcapping=cap),
            get_max_num_attention_heads=lambda: self.heads,
            get_num_kv_heads=lambda *args: self.heads,
        )
        runner = SimpleNamespace(
            model_config=config,
            dtype=self.dtype,
            sliding_window_size=-1,
            req_to_token_pool=SimpleNamespace(req_to_token=None),
        )
        with patch.object(
            backend_module,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
        ):
            prefill = backend_module.FlashInferIndicesUpdaterPrefill(runner, backend)
            decode = backend_module.FlashInferIndicesUpdaterDecode(runner, backend)
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            logit_cap=cap or 0.0,
            tp_q_head_num=self.heads,
            tp_k_head_num=self.heads,
            tp_v_head_num=self.heads,
            head_dim=self.dim,
            scaling=1.0,
            attn_type=0,
            sliding_window_size=-1,
            k_scale_float=1.0,
            v_scale_float=1.0,
        )
        return backend, prefill, decode, layer

    def reference(self, cap):
        scores = self.tensor([0, 2, 10], torch.float32).expand(3, 3)
        if cap:
            scores = cap * torch.tanh(scores / cap)
        mask = torch.ones(3, 3, dtype=torch.bool, device="cuda").tril()
        return scores.masked_fill(~mask, -torch.inf).softmax(-1) @ self.tensor(
            [0, 1, 2], torch.float32
        )

    def check_output(self, output, expected):
        output = output.view(-1, self.heads, self.dim).float()
        torch.testing.assert_close(
            output, expected[:, None, None].expand_as(output), atol=1e-3, rtol=1e-3
        )

    def test_prefill(self):
        for cap in (None, 0.0, 1.0, 2.0):
            for ragged, prefix in ((False, 0), (True, 0), (True, 1)):
                with self.subTest(cap=cap, ragged=ragged, prefix=prefix):
                    backend, updater, _, layer = self.make_backend(cap)
                    paged = backend_module.BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace, "NHD", backend="fa2"
                    )
                    paged_len = prefix if ragged else self.length
                    updater.call_begin_forward(
                        backend.prefill_wrapper_ragged,
                        paged,
                        self.tensor([0]),
                        self.tensor([paged_len]),
                        paged_len,
                        self.tensor([3]),
                        self.tensor([prefix]),
                        None,
                        backend.kv_indptr[0],
                        backend.qo_indptr[0],
                        ragged,
                        None,
                        custom_kv_indices=self.indices[:paged_len],
                    )
                    backend.forward_metadata = SimpleNamespace(
                        prefill_wrappers=[paged],
                        use_ragged=ragged,
                        extend_no_prefix=prefix == 0,
                        multi_item_params=None,
                    )
                    output = backend.forward_extend(
                        self.q[prefix:],
                        self.k[prefix:],
                        self.v[prefix:],
                        layer,
                        SimpleNamespace(out_cache_loc=None),
                        save_kv_cache=False,
                    )
                    self.check_output(output, self.reference(cap)[prefix:])

    def test_decode_and_fast_plan(self):
        for cap in (None, 0.0, 1.0, 2.0):
            for tensor_cores in (False, True):
                with self.subTest(cap=cap, tensor_cores=tensor_cores):
                    backend, _, updater, layer = self.make_backend(cap)
                    wrapper = backend_module.BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace,
                        "NHD",
                        use_tensor_cores=tensor_cores,
                        use_cuda_graph=True,
                        paged_kv_indptr_buffer=self.ptr.clone(),
                        paged_kv_indices_buffer=self.indices.clone(),
                        paged_kv_last_page_len_buffer=self.last_page.clone(),
                    )
                    backend.forward_metadata = SimpleNamespace(
                        decode_wrappers=[wrapper]
                    )
                    for fast in (False, True):
                        if fast:
                            wrapper.begin_forward = partial(
                                backend_module.fast_decode_plan, wrapper
                            )
                        updater.call_begin_forward(
                            wrapper,
                            self.tensor([3]),
                            3,
                            backend.kv_indptr[0],
                            None,
                            SimpleNamespace(
                                kv_indptr=self.ptr, kv_indices=self.indices
                            ),
                            torch.tensor([3], dtype=torch.int32),
                            req_pool_indices=self.tensor([0]),
                        )
                        output = backend.forward_decode(
                            self.q[-1:],
                            None,
                            None,
                            layer,
                            SimpleNamespace(out_cache_loc=None),
                            save_kv_cache=False,
                        )
                        self.check_output(output, self.reference(cap)[-1:])
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = backend.forward_decode(
                            self.q[-1:],
                            None,
                            None,
                            layer,
                            SimpleNamespace(out_cache_loc=None),
                            save_kv_cache=False,
                        )
                    graph.replay()
                    self.check_output(output, self.reference(cap)[-1:])

    def test_fast_prefill_plan(self):
        for cap in (0.0, 1.0):
            with self.subTest(cap=cap):
                backend, updater, _, layer = self.make_backend(cap)
                wrapper = backend_module.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace,
                    "NHD",
                    backend="fa2",
                    use_cuda_graph=True,
                    qo_indptr_buf=self.ptr.clone(),
                    paged_kv_indptr_buf=self.ptr.clone(),
                    paged_kv_indices_buf=self.indices.clone(),
                    paged_kv_last_page_len_buf=self.last_page.clone(),
                )
                updater.call_begin_forward(
                    backend.prefill_wrapper_ragged,
                    wrapper,
                    self.tensor([0]),
                    self.tensor([3]),
                    3,
                    self.tensor([3]),
                    self.tensor([0]),
                    None,
                    backend.kv_indptr[0],
                    backend.qo_indptr[0],
                    False,
                    None,
                    custom_kv_indices=self.indices,
                )
                backend_module.fast_prefill_plan(
                    wrapper,
                    self.ptr,
                    self.ptr,
                    self.indices,
                    self.last_page,
                    self.heads,
                    self.heads,
                    self.dim,
                    1,
                    causal=True,
                    q_data_type=self.dtype,
                    kv_data_type=self.dtype,
                    logits_soft_cap=cap,
                    qo_indptr_host=torch.tensor([0, 3], dtype=torch.int32),
                    kv_indptr_host=torch.tensor([0, 3], dtype=torch.int32),
                    kv_lens_host=torch.tensor([3], dtype=torch.int32),
                    max_q_len=3,
                    max_kv_len=3,
                )
                backend.forward_metadata = SimpleNamespace(
                    prefill_wrappers=[wrapper],
                    use_ragged=False,
                    extend_no_prefix=True,
                    multi_item_params=None,
                )
                output = backend.forward_extend(
                    self.q,
                    self.k,
                    self.v,
                    layer,
                    SimpleNamespace(out_cache_loc=None),
                    save_kv_cache=False,
                )
                self.check_output(output, self.reference(cap))


if __name__ == "__main__":
    unittest.main()
