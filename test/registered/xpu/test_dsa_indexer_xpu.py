"""XPU unit tests for the DSA (Dynamic Sparse Attention) indexer.

Mirrors test/registered/kernels/test_dsa_indexer.py for XPU, covering:
  - Indexer creation and basic forward pass (extend + decode modes)
  - rotate_activation (Hadamard transform, PyTorch-native fallback on XPU)
  - FP8 act_quant dispatch on XPU
  - topk selection (torch.topk fallback on XPU, TOPK_V2 disabled)
  - HybridAttnBackend: init_forward_metadata + get_indexer_metadata routing
  - RotaryEmbedding.forward_xpu with 2D k_rope (DSA indexer single-head key)

NOTE: A full end-to-end GLM5.1 integration test requires the reduced
GlmMoeDsaForCausalLM model which is not publicly available, so it is
not included here. These tests use synthetic tensors and mock runners,
matching the style of the CUDA counterpart.
"""

import unittest
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_xpu_ci

_parallel_override = get_parallel().override(attn_tp_size=1)
_parallel_override.__enter__()

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    BaseIndexerMetadata,
    Indexer,
    rotate_activation,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
)
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")

# Configuration matching GLM5.1 index head dimensions on XPU
DEFAULT_CONFIG = {
    "device": "xpu",
    "dtype": torch.bfloat16,
    "kv_cache_dtype": torch.float8_e4m3fn,
    "context_len": 2048,
    "max_bs": 64,
    "hidden_size": 5120,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "rope_head_dim": 64,
    "index_topk": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "max_position_embeddings": 163840,
    "rope_theta": 10000.0,
    "layer_id": 0,
    "page_size": 128,  # XPU uses page_size=128
}


class MockIndexerMetadata(BaseIndexerMetadata):
    """Minimal mock of BaseIndexerMetadata for XPU testing."""

    def __init__(self, batch_size, seq_lens, device="xpu"):
        self.batch_size = batch_size
        self.seq_lens = seq_lens
        self.device = device

    def get_seqlens_int32(self) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_page_table_64(self) -> torch.Tensor:
        max_seq_len = max(self.seq_lens)
        num_blocks = (max_seq_len + 63) // 64
        page_table = torch.zeros(
            (self.batch_size, num_blocks), dtype=torch.int32, device=self.device
        )
        for i in range(self.batch_size):
            n = (self.seq_lens[i] + 63) // 64
            page_table[i, :n] = torch.arange(n, device=self.device)
        return page_table

    def get_page_table_1(self) -> torch.Tensor:
        max_seq_len = max(self.seq_lens)
        page_table = torch.zeros(
            (self.batch_size, max_seq_len), dtype=torch.int32, device=self.device
        )
        for i in range(self.batch_size):
            n = self.seq_lens[i]
            page_table[i, :n] = torch.arange(n, device=self.device)
        return page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        result = []
        for seq_len in self.seq_lens:
            result.extend(range(1, seq_len + 1))
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ks_list, ke_list = [], []
        k_offset = 0
        for seq_len in self.seq_lens:
            ks = torch.full((seq_len,), k_offset, dtype=torch.int32, device=self.device)
            ke = torch.arange(
                k_offset + 1,
                k_offset + seq_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            ks_list.append(ks)
            ke_list.append(ke)
            k_offset += seq_len
        return torch.cat(ks_list), torch.cat(ke_list)

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device="cpu")

    def get_indexer_seq_len(self) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_dsa_extend_len_cpu(self) -> List[int]:
        return list(self.seq_lens)

    def get_token_to_batch_idx(self) -> torch.Tensor:
        result = []
        for batch_idx, seq_len in enumerate(self.seq_lens):
            result.extend([batch_idx] * seq_len)
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def topk_transform(self, logits, topk, **kwargs):
        return torch.topk(logits, k=topk, dim=-1).indices


class MockModelRunner:
    def __init__(self, config=None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.device = cfg["device"]
        self.config = cfg
        self.dtype = cfg["dtype"]
        self.kv_cache_dtype = cfg["kv_cache_dtype"]
        self.is_hybrid_swa = False

        hf_config = type(
            "HfConfig",
            (),
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "index_topk": cfg["index_topk"],
                "index_head_dim": cfg["index_head_dim"],
                "index_n_heads": cfg["index_n_heads"],
            },
        )()

        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": cfg["context_len"],
                "is_multimodal": False,
                "attention_arch": AttentionArch.MLA,
                "num_attention_heads": 128,
                "kv_lora_rank": cfg["kv_lora_rank"],
                "qk_rope_head_dim": cfg["qk_rope_head_dim"],
                "qk_nope_head_dim": cfg["qk_nope_head_dim"],
                "hf_config": hf_config,
            },
        )()

        self.sliding_window_size = None
        self.page_size = cfg["page_size"]

        max_batch_size = cfg["max_bs"]
        max_context_len = cfg["context_len"]
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_batch_size,
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )()

        self.token_to_kv_pool = DSATokenToKVPool(
            size=max_batch_size * max_context_len,
            page_size=cfg["page_size"],
            dtype=cfg["kv_cache_dtype"],
            kv_lora_rank=cfg["kv_lora_rank"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            layer_num=1,
            device=self.device,
            index_head_dim=cfg["index_head_dim"],
            enable_memory_saver=False,
            kv_cache_dim=cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"],
        )

        # XPU-specific DSA backend settings (mirrors server_args.py XPU section)
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "auto",
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
                "dsa_prefill_backend": "intel_xpu",
                "dsa_decode_backend": "intel_xpu",
                "dsa_topk_backend": "torch",  # XPU uses torch.topk fallback
                "dsa_paged_mqa_logits_backend": "auto",
            },
        )()
        self.hisparse_coordinator = None


@unittest.skipIf(not torch.xpu.is_available(), "XPU is required")
class TestDSAIndexerXPU(CustomTestCase):
    """Tests for the DSA indexer on XPU, mirroring test_dsa_indexer.py."""

    @classmethod
    def setUpClass(cls):
        server_args = ServerArgs(model_path="dummy")
        server_args.enable_dp_attention = False
        server_args.dsa_prefill_backend = "intel_xpu"
        server_args.dsa_decode_backend = "intel_xpu"
        server_args.dsa_topk_backend = "torch"
        # Disable CUDA-only JIT topk-v2 (TileLang requires CUDA_HOME)
        envs.SGLANG_OPT_USE_TOPK_V2.set(False)
        set_global_server_args_for_scheduler(server_args)

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 128
        self.config = DEFAULT_CONFIG.copy()
        self.device = "xpu"
        self.dtype = torch.bfloat16

    def _init_model_runner(self, config_override=None):
        cfg = {**self.config, **(config_override or {})}
        self.model_runner = MockModelRunner(cfg)
        self.backend = DeepseekSparseAttnBackend(self.model_runner)

    def _create_indexer(self, **kwargs):
        params = {
            "hidden_size": self.config["hidden_size"],
            "index_n_heads": self.config["index_n_heads"],
            "index_head_dim": self.config["index_head_dim"],
            "rope_head_dim": self.config["rope_head_dim"],
            "index_topk": self.config["index_topk"],
            "q_lora_rank": self.config["q_lora_rank"],
            "max_position_embeddings": self.config["max_position_embeddings"],
            "rope_theta": self.config["rope_theta"],
            "layer_id": self.config["layer_id"],
            "scale_fmt": "ue8m0",
            "block_size": 128,
            "quant_config": None,
            # GLM5.1 has indexer_rope_interleave=True → is_neox_style=False.
            # The XPU sgl_kernel.rotary_embedding 3D+neox path returns 4D output,
            # so use is_neox_style=False to match the real model config.
            "is_neox_style": False,
        }
        params.update(kwargs)

        torch.set_default_dtype(self.dtype)
        with torch.device(self.device):
            indexer = Indexer(**params)
        indexer = indexer.to(device=self.device)

        for name, module in indexer.named_modules():
            if isinstance(module, LinearBase) and not isinstance(module, LayerNorm):
                if "weights_proj" not in name:
                    module.to(dtype=self.dtype)
        return indexer

    def _create_forward_batch(self, mode, batch_size=None, seq_len=None):
        batch_size = batch_size or self.batch_size
        seq_len = seq_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, seq_len), device=self.device
                ),
                out_cache_loc=torch.arange(batch_size * seq_len, device=self.device),
                seq_lens_sum=batch_size * seq_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor([seq_len] * batch_size, device=self.device),
                seq_lens_cpu=torch.tensor([seq_len] * batch_size, device="cpu"),
                extend_prefix_lens=torch.zeros(
                    batch_size, device=self.device, dtype=torch.int32
                ),
                extend_prefix_lens_cpu=torch.zeros(
                    batch_size, device="cpu", dtype=torch.int32
                ),
                extend_seq_lens=torch.tensor(
                    [seq_len] * batch_size, device=self.device
                ),
                extend_seq_lens_cpu=torch.tensor([seq_len] * batch_size, device="cpu"),
            )
        else:  # DECODE
            total_len = seq_len + 1
            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(0, 100, (batch_size, 1), device=self.device),
                out_cache_loc=torch.arange(
                    batch_size * seq_len, batch_size * total_len, device=self.device
                ),
                seq_lens_sum=batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor([total_len] * batch_size, device=self.device),
                seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
            )

        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            set_forward_context,
        )

        set_forward_context(ForwardContext(attn_backend=self.backend))

        page_size = self.model_runner.page_size
        for i in range(batch_size):
            for j in range(seq_len + (0 if mode == ForwardMode.EXTEND else 1)):
                self.model_runner.req_to_token_pool.req_to_token[i, j] = (
                    i * seq_len + j + page_size
                )
        return forward_batch

    def _verify_topk_output(self, topk_indices, batch_size, q_len, topk):
        self.assertIsNotNone(topk_indices)
        self.assertEqual(topk_indices.device.type, "xpu")
        self.assertEqual(len(topk_indices.shape), 2)
        self.assertEqual(topk_indices.shape[0], batch_size * q_len)
        self.assertGreaterEqual(topk_indices.shape[1], topk)

    # ------------------------------------------------------------------
    # Test: indexer creation
    # ------------------------------------------------------------------

    def test_indexer_basic_creation(self):
        """Test basic Indexer instantiation on XPU."""
        self._init_model_runner()
        indexer = self._create_indexer()

        self.assertEqual(indexer.hidden_size, self.config["hidden_size"])
        self.assertEqual(indexer.n_heads, self.config["index_n_heads"])
        self.assertEqual(indexer.head_dim, self.config["index_head_dim"])
        self.assertEqual(indexer.rope_head_dim, self.config["rope_head_dim"])
        self.assertEqual(indexer.index_topk, self.config["index_topk"])

    # ------------------------------------------------------------------
    # Test: rotate_activation (Hadamard, XPU uses PyTorch-native fallback)
    # ------------------------------------------------------------------

    def test_rotate_activation_power_of_two(self):
        """rotate_activation should work for power-of-2 sizes on XPU (PyTorch fallback)."""
        for hidden_size in [64, 128, 256]:
            x = torch.randn(16, hidden_size, dtype=torch.bfloat16, device=self.device)
            out = rotate_activation(x)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(out.device.type, "xpu")

    def test_rotate_activation_invalid_size(self):
        """rotate_activation should raise for non-power-of-2 sizes."""
        x = torch.randn(16, 129, dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(AssertionError):
            rotate_activation(x)

    # ------------------------------------------------------------------
    # Test: indexer forward — extend mode
    # ------------------------------------------------------------------

    @patch("sglang.srt.hardware_backend.xpu.kernels.dsa.act_quant.act_quant")
    def test_forward_extend_mode(self, mock_act_quant):
        """Indexer forward in EXTEND mode calls sgl_kernel.fp8_mqa_logits on XPU."""

        def _mock_quant(x, block_size=128, scale_fmt=None, *args, **kwargs):
            # Match real act_quant output: scale shape is (*x.shape[:-1], n_groups)
            n_groups = x.shape[-1] // block_size
            scale_shape = x.shape[:-1] + (n_groups,)
            return x.to(torch.float8_e4m3fn), torch.ones(
                scale_shape, dtype=torch.float32, device=x.device
            )

        mock_act_quant.side_effect = _mock_quant

        self._init_model_runner()
        indexer = self._create_indexer()
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND)

        total_tokens = self.batch_size * self.seq_len
        hidden_states = torch.randn(
            total_tokens,
            self.config["hidden_size"],
            dtype=self.dtype,
            device=self.device,
        )
        q_lora = torch.randn(
            total_tokens,
            self.config["q_lora_rank"],
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.arange(total_tokens, device=self.device)

        with patch.object(
            self.backend,
            "get_indexer_metadata",
            return_value=MockIndexerMetadata(
                self.batch_size, [self.seq_len] * self.batch_size, device=self.device
            ),
        ):
            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

        self._verify_topk_output(
            topk_indices, self.batch_size, self.seq_len, self.config["index_topk"]
        )

    # ------------------------------------------------------------------
    # Test: indexer forward — decode mode
    # ------------------------------------------------------------------

    @patch("sglang.srt.hardware_backend.xpu.kernels.dsa.act_quant.act_quant")
    def test_forward_decode_mode(self, mock_act_quant):
        """Indexer forward in DECODE mode calls sgl_kernel.fp8_paged_mqa_logits on XPU."""

        def _mock_quant(x, block_size=128, scale_fmt=None, *args, **kwargs):
            # Match real act_quant output: scale shape is (*x.shape[:-1], n_groups)
            n_groups = x.shape[-1] // block_size
            scale_shape = x.shape[:-1] + (n_groups,)
            return x.to(torch.float8_e4m3fn), torch.ones(
                scale_shape, dtype=torch.float32, device=x.device
            )

        mock_act_quant.side_effect = _mock_quant

        self._init_model_runner()
        indexer = self._create_indexer()
        forward_batch = self._create_forward_batch(ForwardMode.DECODE)

        hidden_states = torch.randn(
            self.batch_size,
            self.config["hidden_size"],
            dtype=self.dtype,
            device=self.device,
        )
        q_lora = torch.randn(
            self.batch_size,
            self.config["q_lora_rank"],
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.arange(self.batch_size, device=self.device)

        with patch.object(
            self.backend,
            "get_indexer_metadata",
            return_value=MockIndexerMetadata(
                self.batch_size,
                [self.seq_len + 1] * self.batch_size,
                device=self.device,
            ),
        ):
            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

        self._verify_topk_output(
            topk_indices, self.batch_size, 1, self.config["index_topk"]
        )

    # ------------------------------------------------------------------
    # Test: skip logits when seq_len <= index_topk
    # ------------------------------------------------------------------

    def test_skip_logits_short_sequence(self):
        """Indexer returns dense topk when seq_len <= index_topk (no FP8 scoring).

        When all KV positions fit within index_topk, the indexer skips the
        expensive FP8 MQA logit computation (EXTEND mode only) and returns
        sequential dense indices covering all KV positions.
        """
        short_seq = self.config["index_topk"] // 2  # 32 < index_topk=64

        self._init_model_runner()
        indexer = self._create_indexer()
        # EXTEND mode is required: _should_skip_logits_computation only triggers
        # for extend (prefill) batches, not decode.
        forward_batch = self._create_forward_batch(
            ForwardMode.EXTEND, seq_len=short_seq
        )

        total_tokens = self.batch_size * short_seq
        hidden_states = torch.randn(
            total_tokens,
            self.config["hidden_size"],
            dtype=self.dtype,
            device=self.device,
        )
        q_lora = torch.randn(
            total_tokens,
            self.config["q_lora_rank"],
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.arange(total_tokens, device=self.device)

        # seq_len (32) < index_topk (64): skip FP8 scoring, use dense fallback.
        # Returns sequential topk indices, NOT None.
        with patch.object(
            self.backend,
            "get_indexer_metadata",
            return_value=MockIndexerMetadata(
                self.batch_size,
                [short_seq] * self.batch_size,
                device=self.device,
            ),
        ):
            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

        # Dense fallback: indices are returned (not None), all within [0, index_topk)
        self.assertIsNotNone(topk_indices)
        self.assertEqual(topk_indices.device.type, "xpu")
        self.assertGreaterEqual(topk_indices.shape[-1], self.config["index_topk"])

    # ------------------------------------------------------------------
    # Test: RotaryEmbedding.forward_xpu with 2D k_rope (DSA indexer path)
    # ------------------------------------------------------------------

    def test_rotary_embedding_2d_key(self):
        """forward_xpu must handle 2D (N, head_size) k_rope from the DSA indexer.

        The DSA indexer creates a RotaryEmbedding with head_size=rope_head_dim
        (64) and a single KV head. Its k_rope tensor is 2D (N, 64), not 3D.
        The XPU fallback path (sgl_kernel.rotary_embedding) requires 3D input;
        forward_xpu must unsqueeze/squeeze transparently.
        """
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        rope_head_dim = self.config["rope_head_dim"]  # 64
        num_tokens = 8
        max_position = self.config["max_position_embeddings"]

        rope = RotaryEmbedding(
            head_size=rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position_embeddings=max_position,
            base=self.config["rope_theta"],
            is_neox_style=False,  # GLM5.1 has indexer_rope_interleave=True → is_neox=False
            dtype=self.dtype,
        ).to(self.device)

        positions = torch.arange(num_tokens, device=self.device)
        # 2D query and key — this is what the DSA indexer passes
        query_2d = torch.randn(
            num_tokens, rope_head_dim, dtype=self.dtype, device=self.device
        )
        key_2d = torch.randn(
            num_tokens, rope_head_dim, dtype=self.dtype, device=self.device
        )

        q_out, k_out = rope.forward_xpu(positions, query_2d, key_2d, rope_head_dim)

        self.assertEqual(q_out.shape, query_2d.shape)
        self.assertEqual(k_out.shape, key_2d.shape)
        self.assertEqual(q_out.device.type, "xpu")

    # ------------------------------------------------------------------
    # Test: XPU uses a single DeepseekSparseAttnBackend for both modes
    # ------------------------------------------------------------------

    def test_unified_dsa_backend_both_modes(self):
        """On XPU, DeepseekSparseAttnBackend handles both prefill and decode.

        Verifies that after _init_model_runner the server_args use the same
        "intel_xpu" impl for both dsa_prefill_backend and dsa_decode_backend,
        matching the unified flash_mla_prefill + flash_mla_decode path in
        sgl-kernel-xpu (no HybridAttnBackend required).
        """
        self._init_model_runner()
        sa = self.model_runner.server_args
        self.assertEqual(sa.dsa_prefill_backend, "intel_xpu")
        self.assertEqual(sa.dsa_decode_backend, "intel_xpu")

        # The backend created for both forward modes should be DeepseekSparseAttnBackend
        backend = self.backend
        self.assertIsInstance(backend, DeepseekSparseAttnBackend)

        # Its prefill impl should also be "intel_xpu"
        self.assertEqual(
            backend.dsa_prefill_impl,
            "intel_xpu",
            "Expected prefill impl 'intel_xpu'; got {}".format(
                backend.dsa_prefill_impl
            ),
        )

    # ------------------------------------------------------------------
    # Test: HybridAttnBackend routes indexer metadata to decode backend
    # ------------------------------------------------------------------

    def test_hybrid_backend_indexer_metadata_routing(self):
        """HybridAttnBackend.get_indexer_metadata delegates to the decode backend.

        Tests the generic HybridAttnBackend routing: when a hybrid setup uses
        DSA for decode and another backend (e.g. Triton) for prefill, the DSA
        decode backend must return indexer metadata even for prefill batches.
        Note: on XPU, GLM5.1 no longer uses HybridAttnBackend (both prefill
        and decode use DeepseekSparseAttnBackend directly), but this routing
        logic remains valid for other hybrid configurations.
        """
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self._init_model_runner()

        # Build a minimal HybridAttnBackend: decode=DSA, prefill=Triton mock
        mock_triton = MagicMock(spec=TritonAttnBackend)
        mock_triton.get_indexer_metadata.return_value = None  # Triton has no indexer

        hybrid = HybridAttnBackend.__new__(HybridAttnBackend)
        hybrid.decode_backend = self.backend
        hybrid.prefill_backend = mock_triton

        # DSA backend should provide metadata for a decode batch
        forward_batch = MagicMock()
        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.batch_size = self.batch_size
        forward_batch.seq_lens = torch.tensor(
            [self.seq_len + 1] * self.batch_size, device=self.device
        )

        # Patch DSA backend's own get_indexer_metadata to return a mock result
        sentinel = object()
        with patch.object(self.backend, "get_indexer_metadata", return_value=sentinel):
            result = hybrid.get_indexer_metadata(
                layer_id=0, forward_batch=forward_batch
            )

        self.assertIs(result, sentinel)
        # Triton backend's get_indexer_metadata should NOT have been called
        mock_triton.get_indexer_metadata.assert_not_called()

    # ------------------------------------------------------------------
    # Test: init_forward_metadata calls both backends for prefill
    # ------------------------------------------------------------------

    def test_hybrid_backend_prefill_initializes_decode_backend(self):
        """HybridAttnBackend.init_forward_metadata calls decode_backend for prefill.

        Tests the generic HybridAttnBackend routing: in a hybrid setup, the
        DSA decode backend must be initialized during prefill so it can manage
        the K-cache. Note: on XPU, GLM5.1 no longer uses HybridAttnBackend
        (both prefill and decode go through DeepseekSparseAttnBackend directly),
        but this routing logic remains valid for other hybrid configurations.
        """
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self._init_model_runner()

        mock_triton = MagicMock(spec=TritonAttnBackend)
        mock_dsa = MagicMock(spec=DeepseekSparseAttnBackend)

        hybrid = HybridAttnBackend.__new__(HybridAttnBackend)
        hybrid.decode_backend = mock_dsa
        hybrid.prefill_backend = mock_triton
        hybrid._select_backend = lambda mode: mock_triton  # Always pick triton

        forward_batch = MagicMock()
        forward_batch.forward_mode = ForwardMode.EXTEND

        hybrid.init_forward_metadata(forward_batch)

        # Both backends must be initialized
        mock_triton.init_forward_metadata.assert_called_once_with(forward_batch)
        mock_dsa.init_forward_metadata.assert_called_once_with(forward_batch)


if __name__ == "__main__":
    unittest.main()
