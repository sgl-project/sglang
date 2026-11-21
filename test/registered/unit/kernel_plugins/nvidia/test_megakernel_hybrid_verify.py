# SPDX-License-Identifier: Apache-2.0
"""Native CPU contracts for the typed MLA/KDA verification handoff."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)
from sglang.srt.mem_cache.kv.decl import STATE_PART
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def fixture():
    # Real layer routing and native state slicing, with small CPU tensor pools.
    mamba = MambaPool.__new__(MambaPool)
    mamba.size = 4
    mamba.mamba_cache = MambaPool.SpeculativeState(
        conv=[torch.zeros(2, 5, 3, 4, dtype=torch.bfloat16)],
        temporal=torch.zeros(2, 5, 2, 4, 4),
        intermediate_conv_window=[torch.zeros(2, 5, 6, 3, 4, dtype=torch.bfloat16)],
        intermediate_ssm=torch.zeros(2, 5, 6, 2, 4, 4),
    )
    req = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    req.state = StateStorage.empty(req.state_sub_pool)
    req.size, req.mamba_pool, req.mamba_map = 4, mamba, {0: 1, 2: 0}
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.use_mla, pool.use_dsa, pool.page_size = True, False, 64
    pool.full_attention_layer_id_mapping = {1: 0}
    pool.full_kv_pool = NS(
        size=128,
        kv_cache_dim=8,
        post_capture_active=False,
        dtype=torch.float8_e4m3fn,
        store_dtype=torch.uint8,
        kv_buffer=[torch.zeros(192, 1, 8, dtype=torch.uint8)],
    )
    pool.layer_transfer_counter = None
    pool.layer_router = KVLayerRouter.from_layer_maps(pool, [])
    pool.layer_router.bind_part(
        sub_pool=req.state_sub_pool,
        part=STATE_PART,
        view=mamba,
        indices=req.mamba_map,
    )
    full = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
    full.forward_decode_metadata = TRTLLMMLADecodeMetadata(
        block_kv_indices=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens_k=torch.tensor([10, 20], dtype=torch.int32),
    )
    full.forward_metadata = None  # Not the TRTLLM decode/verify metadata.
    linear = KDAAttnBackend.__new__(KDAAttnBackend)
    linear.req_to_token_pool, linear.token_to_kv_pool = req, pool
    linear.accept_lens_pool = None
    linear.verify_intermediate_state_indices = torch.tensor(
        [3, 0, 2, 1], dtype=torch.int32
    )
    linear.forward_metadata = NS(
        mamba_cache_indices=torch.tensor([4, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 6, 12], dtype=torch.int32),
        retrieve_parent_token=None,
    )
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend.token_to_kv_pool, backend.req_to_token_pool = pool, req
    backend.full_attn_backend, backend.linear_attn_backend = full, linear
    runner = NS(
        token_to_kv_pool=pool,
        req_to_token_pool=req,
        attn_backend=backend,
        dtype=torch.bfloat16,
        get_aux_hidden_states_width=lambda: 12,
    )
    batch = NS(
        batch_size=2,
        forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=torch.zeros(12, dtype=torch.int64),
        out_cache_loc=torch.arange(12, dtype=torch.int64),
        req_pool_indices=torch.tensor([2, 4], dtype=torch.int32),
        spec_info=NS(draft_token_num=6),
        ragged_verify_layout=None,
        aux_hidden_states_buffer=None,
    )
    binding = mb.MegakernelBinding(model=torch.nn.Module())
    return NS(
        binding=binding,
        runner=runner,
        batch=batch,
        mamba=mamba,
        pool=pool,
        linear=linear,
        full=full,
        backend=backend,
    )


class TestHybridVerify(unittest.TestCase):
    def setUp(self):
        self.f = fixture()
        self.env = patch.dict("os.environ", {"SGLANG_ENABLE_PP_SPEC": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.f.prepared = self.f.binding.preparation_view(self.f.runner)

    def view(self, *, capturing=False):
        with (
            patch.object(mb, "get_attn_backend", return_value=self.f.backend),
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=capturing,
            ),
        ):
            return self.f.binding.forward_view(self.f.batch).hybrid

    def test_native_mapping_borrowed_checkpoints_and_replacement_detection(self):
        f = self.f
        for layer in (0, 2):
            expected = f.linear.state_layer_cache(layer)
            actual = f.prepared.hybrid.kda.state(layer)
            for lhs, rhs in (
                (actual.conv[0], expected.conv[0]),
                (actual.intermediate_ssm, expected.intermediate_ssm),
                (
                    actual.intermediate_conv_window[0],
                    expected.intermediate_conv_window[0],
                ),
            ):
                self.assertEqual(mb._identity(lhs), mb._identity(rhs))
        f.binding.assert_live()
        f.mamba.mamba_cache.intermediate_ssm.add_(1)  # Values may change on replay.
        f.binding.assert_live()
        from dataclasses import replace

        f.mamba.mamba_cache = replace(
            f.mamba.mamba_cache,
            intermediate_ssm=f.mamba.mamba_cache.intermediate_ssm.clone(),
        )
        with self.assertRaisesRegex(RuntimeError, "hybrid buffers changed"):
            f.binding.assert_live()

    def test_q6_native_metadata_and_eager_aux_allocation(self):
        view = self.view()
        f = self.f
        self.assertEqual(view.mla.tokens_per_request, 6)
        self.assertIs(
            view.mla.block_table, f.full.forward_decode_metadata.block_kv_indices
        )
        self.assertIs(
            view.kda.state_indices, f.linear.forward_metadata.mamba_cache_indices
        )
        self.assertEqual(view.kda.verify_scratch_indices.tolist(), [3, 0])
        self.assertEqual(
            view.kda.verify_scratch_indices.data_ptr(),
            f.linear.verify_intermediate_state_indices.data_ptr(),
        )
        self.assertEqual(view.aux_hidden_states.shape, (12, 12))
        self.assertIs(view.aux_hidden_states, f.batch.aux_hidden_states_buffer)
        self.assertIs(self.view().aux_hidden_states, view.aux_hidden_states)

    def test_capture_reuses_runner_aux_and_seals_owners(self):
        f = self.f
        with self.assertRaisesRegex(RuntimeError, "runner auxiliary buffer"):
            self.view(capturing=True)
        owner = torch.empty(24, 12, dtype=torch.bfloat16)
        f.batch.aux_hidden_states_buffer = owner[:12]
        with patch.object(
            torch, "empty", side_effect=AssertionError("capture allocated")
        ):
            view = self.view(capturing=True)
        f.binding.check_capture_aux(NS(buffers=NS(aux_hidden_states=owner)))
        with patch.object(mb, "get_attn_backend", return_value=f.backend):
            f.binding.assert_live()
            f.batch.aux_hidden_states_buffer = view.aux_hidden_states.clone()
            with self.assertRaisesRegex(RuntimeError, "captured verify.*aux"):
                f.binding.assert_live()
        with self.assertRaisesRegex(RuntimeError, "differs from runner"):
            f.binding.check_capture_aux(NS(buffers=NS(aux_hidden_states=owner.clone())))

    def test_unsupported_verify_metadata_and_ownership_fail_closed(self):
        for obj, field, value, message in (
            (self.f.batch, "ragged_verify_layout", object(), "uniform"),
            (
                self.f.linear.forward_metadata,
                "retrieve_parent_token",
                torch.zeros(1),
                "uniform",
            ),
            (self.f.batch, "input_ids", torch.zeros(11), "request/token"),
            (self.f.linear, "accept_lens_pool", torch.ones(1), "fused acceptance"),
        ):
            with (
                self.subTest(field=field),
                patch.object(obj, field, value),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.view()
        with (
            patch.dict("os.environ", {"SGLANG_ENABLE_PP_SPEC": "1"}),
            self.assertRaisesRegex(ValueError, "PP-stable"),
        ):
            self.view()
        self.f.linear.verify_intermediate_state_indices = torch.arange(
            4, dtype=torch.int32
        )
        with self.assertRaisesRegex(RuntimeError, "scratch indices changed"):
            self.f.binding.assert_live()

    def test_decode_defaults_do_not_expose_verify_resources(self):
        f = self.f
        f.batch.forward_mode = ForwardMode.DECODE
        view = self.view()
        self.assertEqual(view.mla.tokens_per_request, 1)
        self.assertIsNone(view.kda.verify_scratch_indices)
        self.assertIsNone(view.aux_hidden_states)


if __name__ == "__main__":
    unittest.main()
