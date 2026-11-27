"""CPU checks of the megakernel binding: the view is the running forward's attention
backend and KV pool, and a source replaced after capture is caught."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class Pool:
    uniform_fp8 = False
    request_window = None

    def __init__(self):
        self.keys = [torch.zeros(4, 8) for _ in range(2)]

    def get_key_buffer(self, layer_id):
        return self.keys[layer_id]

    def get_swa_key_buffer_radix(self, layer_id):
        return self.keys[layer_id]

    def get_extra_key_page_size(self, layer_id):
        return 64 + layer_id

    def set_swa_key_buffer_radix_fused_norm_rope(self, **kw):
        return kw


def backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(core_attn_metadata="core"),
        trtllm_attn=False,
        forward=lambda *a, **k: "attention",
        store_cache=lambda **k: None,
        get_swa_out_cache_loc=lambda fb: "swa_loc",
        forward_low_ratio_sources=lambda **k: "sources",
        forward_core_compressor=lambda **k: "compressed",
    )


class TestMegakernelBinding(CustomTestCase):
    def test_the_view_is_the_running_forwards_backend_and_pool(self):
        pool = Pool()
        with patch.object(mb, "get_attn_backend", return_value=backend(pool)):
            view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        self.assertEqual(view.metadata.core_attn_metadata, "core")
        self.assertIs(view.kv_pool.pool, pool)
        self.assertIs(view.kv_pool.get_swa_key_buffer_radix(1), pool.keys[1])

    def test_a_replaced_source_fails_assert_live_by_name(self):
        model = torch.nn.Module()
        model.engram_hasher = SimpleNamespace(history=torch.zeros(5, 3))
        binding = mb.MegakernelBinding(model=model)
        attn_backend = backend(Pool())
        with patch.object(mb, "get_attn_backend", return_value=attn_backend):
            binding.forward_view()
            self.assertIs(binding.engram_history(), model.engram_hasher.history)
            binding.assert_live()
            model.engram_hasher.history = torch.zeros(9, 3)
            with self.assertRaisesRegex(RuntimeError, "engram_history"):
                binding.assert_live()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        with patch.object(mb, "get_attn_backend", return_value=backend(Pool())):
            binding.forward_view()
        with patch.object(mb, "get_attn_backend", return_value=backend(Pool())):
            with self.assertRaisesRegex(RuntimeError, "kv_pool"):
                binding.assert_live()

    def test_mla_cache_is_a_view_of_the_native_pool(self):
        pool = Pool()
        pool.keys[1] = torch.empty((8, 1, 576), dtype=torch.float8_e4m3fn)
        attn_backend = backend(pool)
        attn_backend.use_fused_topk = True
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        with patch.object(mb, "get_attn_backend", return_value=attn_backend):
            view = binding.forward_view()
            cache = view.kv_pool.get_key_buffer(1)
            self.assertIs(cache, pool.keys[1])
            self.assertEqual(cache.dtype, torch.float8_e4m3fn)
            self.assertEqual(cache.stride(), (576, 576, 1))
            self.assertTrue(view.use_fused_topk)
            binding.assert_live()
            self.assertIs(binding.forward_view().kv_pool.get_key_buffer(1), cache)
            pool.keys[1] = torch.empty_like(cache)
            with self.assertRaisesRegex(RuntimeError, r"kv_cache\[1\]"):
                binding.assert_live()
            with self.assertRaisesRegex(RuntimeError, r"kv_cache\[1\]"):
                binding.forward_view().kv_pool.get_key_buffer(1)

    def test_mla_cache_stride_change_is_detected(self):
        pool = Pool()
        storage = torch.empty((8, 1, 576), dtype=torch.float8_e4m3fn)
        pool.keys[1] = storage[::2]
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        with patch.object(mb, "get_attn_backend", return_value=backend(pool)):
            view = binding.forward_view()
            self.assertFalse(view.use_fused_topk)
            view.kv_pool.get_key_buffer(1)
            pool.keys[1] = storage[:4]
            with self.assertRaisesRegex(RuntimeError, r"kv_cache\[1\]"):
                binding.assert_live()


class DsaPool:
    index_page_size = 64

    def __init__(self):
        self.keys = [torch.zeros(4, 1, 8) for _ in range(2)]
        self.index_k = [torch.zeros(4, 12) for _ in range(2)]

    def get_key_buffer(self, layer_id):
        return self.keys[layer_id]

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.index_k[layer_id]

    def set_mla_kv_buffer(self, layer, loc, k, k_rope):
        return layer, loc, k, k_rope


def dsa_backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata="metadata",
        dsa_decode_impl="trtllm",
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        kv_cache_dim=656,
        physical_page_size=64,
        use_mha=False,
        kv_cache_dtype=torch.float8_e4m3fn,
        use_fused_topk=True,
        dsa_topk_backend="topk backend",
        attn_dcp_size=1,
        dsa_index_kpool=1,
        hisparse_coordinator=None,
        workspace_buffer=torch.zeros(16, dtype=torch.uint8),
        _multi_ctas_kv_counter_buffer=torch.zeros(8, dtype=torch.uint8),
        get_indexer_metadata=lambda layer_id, fb: ("indexer", layer_id, fb),
    )


class TestMegakernelBindingDsa(CustomTestCase):
    def test_the_dsa_view_is_the_running_dsa_backend(self):
        backend = dsa_backend(DsaPool())
        with patch.object(mb, "get_attn_backend", return_value=backend):
            dsa = mb.MegakernelBinding(model=torch.nn.Module()).forward_view().dsa
        same = {
            "kv_lora_rank": "kv_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "qk_rope_head_dim": "qk_rope_head_dim",
            "kv_cache_dim": "kv_cache_dim",
            "physical_page_size": "physical_page_size",
            "decode_impl": "dsa_decode_impl",
            "use_mha": "use_mha",
            "kv_cache_dtype": "kv_cache_dtype",
            "use_fused_topk": "use_fused_topk",
            "topk_backend": "dsa_topk_backend",
            "dcp_size": "attn_dcp_size",
            "index_kpool": "dsa_index_kpool",
            "workspace_buffer": "workspace_buffer",
            "multi_ctas_kv_counter_buffer": "_multi_ctas_kv_counter_buffer",
            "indexer_metadata": "get_indexer_metadata",
        }
        for member, attr in same.items():
            self.assertIs(getattr(dsa, member), getattr(backend, attr), member)
        self.assertFalse(dsa.hisparse)
        backend.use_mha = True  # read live: the backend sets it per forward
        self.assertTrue(dsa.use_mha)
        self.assertEqual(dsa.indexer_metadata(3, "fb"), ("indexer", 3, "fb"))

    def test_a_backend_without_dsa_has_no_dsa_view(self):
        backend = SimpleNamespace(token_to_kv_pool=DsaPool(), forward_metadata=None)
        with patch.object(mb, "get_attn_backend", return_value=backend):
            binding = mb.MegakernelBinding(model=torch.nn.Module())
            self.assertIsNone(binding.forward_view().dsa)
            binding.assert_live()

    def test_the_pool_view_offers_the_mla_and_index_k_buffers(self):
        pool = DsaPool()
        view = mb.KvPoolView(pool)
        self.assertIs(view.get_key_buffer(1), pool.keys[1])
        self.assertIs(view.get_index_k_with_scale_buffer(0), pool.index_k[0])
        self.assertEqual(view.index_page_size, 64)
        self.assertEqual(view.set_mla_kv_buffer(1, 2, 3, 4), (1, 2, 3, 4))
        # a pool without index-K pages
        self.assertIsNone(mb.KvPoolView(SimpleNamespace()).index_page_size)

    def test_a_replaced_dsa_buffer_fails_assert_live(self):
        for attr, name in (
            ("workspace_buffer", "dsa_workspace"),
            ("_multi_ctas_kv_counter_buffer", "dsa_kv_counter"),
        ):
            with self.subTest(buffer=attr):
                backend = dsa_backend(DsaPool())
                binding = mb.MegakernelBinding(model=torch.nn.Module())
                with patch.object(mb, "get_attn_backend", return_value=backend):
                    binding.forward_view()
                    binding.assert_live()
                    setattr(backend, attr, torch.zeros(32, dtype=torch.uint8))
                    with self.assertRaisesRegex(RuntimeError, name):
                        binding.assert_live()

    def test_each_draft_step_backend_seals_its_own_dsa_buffers(self):
        """An EAGLE draft's per-step DSA backends each own a KV counter: a later step's
        view is not a reallocation, and a counter regrown after its first read fails
        assert_live whichever backend is running."""
        pool, workspace = DsaPool(), torch.zeros(16, dtype=torch.uint8)
        steps = [dsa_backend(pool) for _ in range(3)]
        for backend in steps:
            # the workspace is one named runtime buffer; the counter is per backend
            backend.workspace_buffer = workspace
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        for backend in steps + steps:
            with patch.object(mb, "get_attn_backend", return_value=backend):
                binding.forward_view()
        with patch.object(mb, "get_attn_backend", return_value=steps[0]):
            binding.assert_live()
            steps[2]._multi_ctas_kv_counter_buffer = torch.zeros(16, dtype=torch.uint8)
            with self.assertRaisesRegex(RuntimeError, r"dsa_kv_counter\[2\]"):
                binding.assert_live()


def hybrid_pools(verify: bool):
    """An MLA + KDA hybrid pool pair with one layer of each; ``verify`` gives the KDA
    layer its target-verify scratch."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MambaPool

    full = SimpleNamespace(
        kv_buffer=[torch.zeros(4 * 2, 3)],
        kv_cache_dim=3,
        size=6,
        post_capture_active=False,
        dtype=torch.bfloat16,
        store_dtype=torch.bfloat16,
    )
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.__dict__.update(
        full_kv_pool=full,
        full_attention_layer_id_mapping={0: 0},
        page_size=2,
        use_mla=True,
        use_dsa=False,
    )
    scratch = {
        "intermediate_conv_window": [torch.zeros(1, 4, 2)],
        "intermediate_ssm": torch.zeros(1, 4, 8, 2),
    }
    state = (
        MambaPool.SpeculativeState(
            conv=[torch.zeros(1, 4, 2)], temporal=torch.zeros(1, 4, 2), **scratch
        )
        if verify
        else MambaPool.State(conv=[torch.zeros(1, 4, 2)], temporal=torch.zeros(1, 4, 2))
    )
    mamba_pool = SimpleNamespace(
        size=3, mamba2_layer_cache=lambda index: state.at_layer_idx(index)
    )
    req_pool = SimpleNamespace(mamba_map={1: 0}, mamba_pool=mamba_pool, size=3)
    return pool, req_pool, state


def hybrid_backend(req_pool, *, verify_rows=None):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend.__dict__.update(
        token_to_kv_pool=None,
        full_attn_backend=SimpleNamespace(
            forward_metadata=SimpleNamespace(
                block_kv_indices=torch.zeros(2, 4, dtype=torch.int32),
                seq_lens_k=torch.ones(2, dtype=torch.int32),
            )
        ),
        linear_attn_backend=SimpleNamespace(
            forward_metadata=SimpleNamespace(
                mamba_cache_indices=torch.tensor([1, 2], dtype=torch.int32)
            ),
            verify_intermediate_state_indices=verify_rows,
            req_to_token_pool=req_pool,
        ),
    )
    return backend


def step_batch(*, verify: bool, aux=None):
    mode = SimpleNamespace(is_target_verify=lambda: verify)
    rows = 2 * (6 if verify else 1)
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=2,
        out_cache_loc=torch.arange(rows),
        input_ids=torch.zeros(rows, dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2]),
        spec_info=SimpleNamespace(draft_token_num=6) if verify else None,
        aux_hidden_states_buffer=aux,
    )


class TestHybridStepViews(CustomTestCase):
    """The hybrid MLA + KDA views: a target verify carries its rows per request, each
    request's verify scratch row and the auxiliary hidden-state rows; decode does not."""

    def _binding(self, *, aux_width=0):
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        binding._aux_width, binding._aux_dtype = aux_width, torch.bfloat16
        return binding

    def test_the_pool_view_offers_the_verify_scratch_only_where_the_pool_keeps_it(self):
        for verify in (False, True):
            with self.subTest(verify=verify):
                pool, req_pool, state = hybrid_pools(verify)
                view = self._binding().pool_view(pool, req_pool).hybrid
                kda = view.kda.state(1)
                if verify:
                    self.assertEqual(
                        kda.intermediate_ssm.data_ptr(),
                        state.intermediate_ssm[0].data_ptr(),
                    )
                    self.assertEqual(
                        kda.intermediate_conv_window[0].data_ptr(),
                        state.intermediate_conv_window[0][0].data_ptr(),
                    )
                else:
                    self.assertEqual(kda.intermediate_conv_window, ())
                    self.assertIsNone(kda.intermediate_ssm)

    def test_a_replaced_verify_scratch_fails_assert_live(self):
        pool, req_pool, state = hybrid_pools(verify=True)
        binding = self._binding()
        binding.pool_view(pool, req_pool)
        backend = hybrid_backend(req_pool)
        with patch.object(mb, "get_attn_backend", return_value=backend):
            binding.assert_live()
            object.__setattr__(state, "intermediate_ssm", torch.zeros(1, 4, 8, 2))
            with self.assertRaisesRegex(RuntimeError, "hybrid buffers changed"):
                binding.assert_live()

    def test_a_decode_step_has_no_verify_fields(self):
        _, req_pool, _ = hybrid_pools(verify=True)
        backend = hybrid_backend(
            req_pool, verify_rows=torch.arange(4, dtype=torch.int32)
        )
        with patch.object(mb, "get_attn_backend", return_value=backend):
            hybrid = self._binding(aux_width=8).forward_view(step_batch(verify=False))
        self.assertEqual(hybrid.hybrid.mla.tokens_per_request, 1)
        self.assertIsNone(hybrid.hybrid.kda.verify_scratch_indices)
        self.assertIsNone(hybrid.hybrid.aux_hidden_states)

    def test_a_target_verify_offers_its_rows_scratch_and_aux_buffer(self):
        _, req_pool, _ = hybrid_pools(verify=True)
        rows = torch.arange(4, dtype=torch.int32)
        backend = hybrid_backend(req_pool, verify_rows=rows)
        aux = torch.zeros(12, 8, dtype=torch.bfloat16)
        with patch.object(mb, "get_attn_backend", return_value=backend):
            view = self._binding(aux_width=8).forward_view(
                step_batch(verify=True, aux=aux)
            )
        self.assertEqual(view.hybrid.mla.tokens_per_request, 6)
        self.assertTrue(torch.equal(view.hybrid.kda.verify_scratch_indices, rows[:2]))
        self.assertIs(view.hybrid.aux_hidden_states, aux)

    def test_an_eager_verify_gets_an_aux_buffer_and_a_captured_one_must_have_one(self):
        _, req_pool, _ = hybrid_pools(verify=True)
        backend = hybrid_backend(
            req_pool, verify_rows=torch.arange(4, dtype=torch.int32)
        )
        binding = self._binding(aux_width=8)
        with (
            patch.object(mb, "get_attn_backend", return_value=backend),
            patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
        ):
            batch = step_batch(verify=True)
            view = binding.forward_view(batch)
            aux = view.hybrid.aux_hidden_states
            self.assertEqual(tuple(aux.shape), (12, 8))
            self.assertEqual(aux.dtype, torch.bfloat16)
            self.assertIs(batch.aux_hidden_states_buffer, aux)
            with patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
                with self.assertRaisesRegex(RuntimeError, "auxiliary hidden-state"):
                    binding.forward_view(step_batch(verify=True))

    def test_a_target_without_aux_taps_offers_none(self):
        _, req_pool, _ = hybrid_pools(verify=True)
        backend = hybrid_backend(
            req_pool, verify_rows=torch.arange(4, dtype=torch.int32)
        )
        with patch.object(mb, "get_attn_backend", return_value=backend):
            view = self._binding(aux_width=0).forward_view(step_batch(verify=True))
        self.assertIsNone(view.hybrid.aux_hidden_states)


if __name__ == "__main__":
    unittest.main()
