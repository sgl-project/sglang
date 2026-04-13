"""
Tests for TRTLLM MHA attention backend.

- test_decode_output_match / test_extend_output_match:
    Compare TRTLLM MHA vs FlashInfer reference across kv_cache_dtype variants.
- test_rope_fusion_decode_output_match / test_rope_fusion_extend_output_match:
    Compare fused (RoPE+FP8 quant+KV cache) vs unfused path.
"""

import os
import unittest

import torch

from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

DEFAULT_CONFIG = {
    "device": "cuda",
    "dtype": torch.bfloat16,
    "kv_cache_dtype": torch.bfloat16,
    "context_len": 4096,
    "max_bs": 64,
    "page_size": 64,
    "num_attention_heads": 64,
    "num_kv_heads": 8,
    "head_dim": 64,
    "hidden_size": 2880,
    "layer_num": 2,
    "layer_id": 0,
    "rope_theta": 150000,
    "architectures": ["GptOssForCausalLM"],
    "seed": 42,
    "rtol": 0.01,
    "atol": 0.01,
}

TEST_CASES = {
    "basic_decode": [
        {
            "name": "bf16_kv_cache",
            "kv_cache_dtype": torch.bfloat16,
            "batch_size": 4,
            "max_seq_len": 128,
        },
        {
            "name": "fp8_kv_cache",
            "kv_cache_dtype": torch.float8_e4m3fn,
            "batch_size": 4,
            "max_seq_len": 128,
            "atol": 0.05,
        },
    ],
    "basic_extend": [
        {
            "name": "bf16_kv_cache",
            "kv_cache_dtype": torch.bfloat16,
            "seq_lens_list": [64, 100, 80],
        },
        {
            "name": "fp8_kv_cache",
            "kv_cache_dtype": torch.float8_e4m3fn,
            "seq_lens_list": [64, 100, 80],
            "atol": 0.25,
        },
    ],
    "rope_fusion_decode": [
        {
            "name": "rope_fusion",
            "kv_cache_dtype": torch.float8_e4m3fn,
            "batch_size": 4,
            "max_seq_len": 128,
            "atol": 0.05,
        },
    ],
    "rope_fusion_extend": [
        {
            "name": "rope_fusion",
            "kv_cache_dtype": torch.float8_e4m3fn,
            "seq_lens_list": [64, 100, 80],
            "atol": 0.15,
        },
    ],
}


class MockModelRunner:
    """Minimal ModelRunner for testing MHA backends."""

    def __init__(self, config, enable_rope_fusion=False):
        self.device = config["device"]
        self.dtype = config["dtype"]
        self.kv_cache_dtype = config["kv_cache_dtype"]
        self.page_size = config["page_size"]
        self.sliding_window_size = None

        server_args = ServerArgs(model_path="dummy")
        server_args.enable_dp_attention = False
        server_args.disable_piecewise_cuda_graph = False
        server_args.piecewise_cuda_graph_tokens = [4, 8, 16, 32, 64, 128, 256, 512]
        set_global_server_args_for_scheduler(server_args)
        self.server_args = server_args

        if enable_rope_fusion:
            os.environ["SGLANG_ENABLE_FLASHINFER_ROPE_FUSION"] = "1"
        else:
            os.environ["SGLANG_ENABLE_FLASHINFER_ROPE_FUSION"] = "0"

        hf_config = type("HFConfig", (), {"architectures": config["architectures"]})
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": config["context_len"],
                "num_attention_heads": config["num_attention_heads"],
                "hidden_size": config["hidden_size"],
                "head_dim": config["head_dim"],
                "get_num_kv_heads": staticmethod(lambda _: config["num_kv_heads"]),
                "is_multimodal": False,
                "is_encoder_decoder": False,
                "hf_config": hf_config,
            },
        )

        max_bs = config["max_bs"]
        max_ctx = config["context_len"]
        req_to_token = torch.arange(
            max_bs * max_ctx, dtype=torch.int32, device=self.device
        ).reshape(max_bs, max_ctx)
        self.req_to_token_pool = type(
            "TokenPool", (), {"size": max_bs, "req_to_token": req_to_token}
        )

        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_bs * max_ctx,
            page_size=config["page_size"],
            dtype=config["kv_cache_dtype"],
            head_num=config["num_kv_heads"],
            head_dim=config["head_dim"],
            layer_num=config["layer_num"],
            device=config["device"],
            enable_memory_saver=False,
        )

        self.token_to_kv_pool_allocator = type(
            "MockAllocator",
            (),
            {"get_kvcache": lambda self_: self.token_to_kv_pool},
        )()


def _create_layer(config):
    return RadixAttention(
        num_heads=config["num_attention_heads"],
        head_dim=config["head_dim"],
        scaling=1.0 / (config["head_dim"] ** 0.5),
        num_kv_heads=config["num_kv_heads"],
        layer_id=config["layer_id"],
        v_head_dim=config["head_dim"],
        prefix="test_attn",
    )


def _create_rotary_emb(config):
    from sglang.srt.layers.rotary_embedding import get_rope_wrapper

    rotary = get_rope_wrapper(
        head_size=config["head_dim"],
        rotary_dim=config["head_dim"],
        max_position=config["context_len"],
        base=config.get("rope_theta", 10000),
        is_neox_style=True,
        device=config["device"],
    )
    rotary.cos_sin_cache = rotary.cos_sin_cache.to(config["device"])
    return rotary


def _populate_kv_cache(batch_size, seq_lens, model_runner, layer, config):
    torch.manual_seed(config["seed"])
    for b in range(batch_size):
        sl = int(seq_lens[b].item())
        for t in range(sl - 1):
            cache_k = torch.randn(
                1,
                config["num_kv_heads"],
                config["head_dim"],
                dtype=config["dtype"],
                device=config["device"],
            )
            cache_v = torch.randn(
                1,
                config["num_kv_heads"],
                config["head_dim"],
                dtype=config["dtype"],
                device=config["device"],
            )
            loc = model_runner.req_to_token_pool.req_to_token[b, t].unsqueeze(0).long()
            model_runner.token_to_kv_pool.set_kv_buffer(layer, loc, cache_k, cache_v)


def _create_decode_forward_batch(batch_size, seq_lens, backend, model_runner, config):
    out_cache_loc = torch.tensor(
        [
            model_runner.req_to_token_pool.req_to_token[b, int(seq_lens[b].item()) - 1]
            for b in range(batch_size)
        ],
        dtype=torch.int64,
        device=config["device"],
    )
    fb = ForwardBatch(
        batch_size=batch_size,
        input_ids=torch.zeros(batch_size, dtype=torch.int64, device=config["device"]),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens.sum().item()),
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.arange(batch_size, device=config["device"]),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        attn_backend=backend,
    )
    fb.req_to_token_pool = model_runner.req_to_token_pool
    fb.token_to_kv_pool = model_runner.token_to_kv_pool
    fb.positions = (seq_lens - 1).to(torch.int64)
    return fb


def _create_extend_forward_batch(batch_size, seq_lens, backend, model_runner, config):
    total_tokens = int(seq_lens.sum().item())
    out_cache_loc = torch.cat(
        [
            model_runner.req_to_token_pool.req_to_token[b, : int(seq_lens[b].item())]
            for b in range(batch_size)
        ]
    ).to(torch.int64)

    fb = ForwardBatch(
        batch_size=batch_size,
        input_ids=torch.zeros(total_tokens, dtype=torch.int64, device=config["device"]),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens.sum().item()),
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.arange(batch_size, device=config["device"]),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        extend_num_tokens=total_tokens,
        extend_seq_lens=seq_lens.clone(),
        extend_seq_lens_cpu=seq_lens.cpu().tolist(),
        extend_prefix_lens=torch.zeros(
            batch_size, dtype=torch.int32, device=config["device"]
        ),
        extend_prefix_lens_cpu=[0] * batch_size,
        attn_backend=backend,
    )
    fb.req_to_token_pool = model_runner.req_to_token_pool
    fb.token_to_kv_pool = model_runner.token_to_kv_pool
    fb.positions = torch.cat(
        [torch.arange(s, dtype=torch.int64, device=config["device"]) for s in seq_lens]
    )
    return fb


def _compare_outputs(test_case, out_a, out_b, rtol, atol, label=""):
    test_case.assertEqual(out_a.shape, out_b.shape)
    test_case.assertFalse(torch.isnan(out_a).any(), f"{label} output A has NaN")
    test_case.assertFalse(torch.isnan(out_b).any(), f"{label} output B has NaN")

    diff = (out_a.float() - out_b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    test_case.assertTrue(
        torch.allclose(out_a.float(), out_b.float(), rtol=rtol, atol=atol),
        f"{label} outputs differ: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}",
    )


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer required",
)
class TestTRTLLMMHA(CustomTestCase):
    """Test suite for TRTLLM MHA backend."""

    def _merge_config(self, overrides):
        config = DEFAULT_CONFIG.copy()
        config.update(overrides)
        return config

    def _build_trtllm_backend(self, config, enable_rope_fusion=False):
        model_runner = MockModelRunner(config, enable_rope_fusion=enable_rope_fusion)
        backend = TRTLLMHAAttnBackend(model_runner, skip_prefill=True)
        return backend, model_runner

    def _build_reference_backend(self, config):
        model_runner = MockModelRunner(config)
        backend = FlashInferAttnBackend(model_runner)
        return backend, model_runner

    # ------------------------------------------------------------------ #
    #  Fundamental: TRTLLM MHA vs FlashInfer reference                   #
    # ------------------------------------------------------------------ #

    def test_basic_decode_output_match(self):
        """TRTLLM MHA decode should match FlashInfer decode output."""
        for tc in TEST_CASES["basic_decode"]:
            with self.subTest(name=tc["name"]):
                config = self._merge_config(tc)
                bs = config["batch_size"]
                max_seq_len = config["max_seq_len"]
                num_q, num_kv, hdim = (
                    config["num_attention_heads"],
                    config["num_kv_heads"],
                    config["head_dim"],
                )

                torch.manual_seed(config["seed"])
                seq_lens = torch.randint(
                    max_seq_len // 2, max_seq_len + 1, (bs,), device=config["device"]
                )
                q = torch.randn(
                    bs, num_q * hdim, dtype=config["dtype"], device=config["device"]
                )
                k = torch.randn(
                    bs, num_kv * hdim, dtype=config["dtype"], device=config["device"]
                )
                v = torch.randn(
                    bs, num_kv * hdim, dtype=config["dtype"], device=config["device"]
                )

                def run(build_fn):
                    backend, model_runner = build_fn(config)
                    layer = _create_layer(config)
                    _populate_kv_cache(bs, seq_lens, model_runner, layer, config)
                    forward_batch = _create_decode_forward_batch(
                        bs, seq_lens, backend, model_runner, config
                    )
                    backend.init_forward_metadata(forward_batch)
                    return backend.forward_decode(
                        q.clone(), k.clone(), v.clone(), layer, forward_batch
                    )

                out_trtllm = run(self._build_trtllm_backend)
                out_ref = run(self._build_reference_backend)
                _compare_outputs(
                    self,
                    out_trtllm,
                    out_ref,
                    rtol=config["rtol"],
                    atol=config["atol"],
                    label=f"[basic_decode/{config['name']}]",
                )

    def test_basic_extend_output_match(self):
        """TRTLLM MHA extend should match FlashInfer extend output."""
        for tc in TEST_CASES["basic_extend"]:
            with self.subTest(name=tc["name"]):
                config = self._merge_config(tc)
                seq_lens_list = config["seq_lens_list"]
                bs = len(seq_lens_list)
                total_num_tokens = sum(seq_lens_list)
                num_q, num_kv, hdim = (
                    config["num_attention_heads"],
                    config["num_kv_heads"],
                    config["head_dim"],
                )

                torch.manual_seed(config["seed"])
                seq_lens = torch.tensor(
                    seq_lens_list, dtype=torch.int32, device=config["device"]
                )
                q = torch.randn(
                    total_num_tokens,
                    num_q * hdim,
                    dtype=config["dtype"],
                    device=config["device"],
                )
                k = torch.randn(
                    total_num_tokens,
                    num_kv * hdim,
                    dtype=config["dtype"],
                    device=config["device"],
                )
                v = torch.randn(
                    total_num_tokens,
                    num_kv * hdim,
                    dtype=config["dtype"],
                    device=config["device"],
                )

                def run(build_fn):
                    backend, model_runner = build_fn(config)
                    layer = _create_layer(config)
                    forward_batch = _create_extend_forward_batch(
                        bs, seq_lens, backend, model_runner, config
                    )
                    backend.init_forward_metadata(forward_batch)
                    return backend.forward_extend(
                        q.clone(), k.clone(), v.clone(), layer, forward_batch
                    )

                out_trtllm = run(self._build_trtllm_backend)
                out_ref = run(self._build_reference_backend)
                _compare_outputs(
                    self,
                    out_trtllm,
                    out_ref,
                    rtol=config["rtol"],
                    atol=config["atol"],
                    label=f"[basic_extend/{config['name']}]",
                )

    # ------------------------------------------------------------------ #
    #  Rope fusion: fused vs unfused path                                #
    # ------------------------------------------------------------------ #

    def test_rope_fusion_decode_output_match(self):
        """Fused vs unfused decode should produce the same attention output."""
        for tc in TEST_CASES["rope_fusion_decode"]:
            with self.subTest(name=tc["name"]):
                config = self._merge_config(tc)
                bs = config["batch_size"]
                max_seq_len = config["max_seq_len"]
                num_q, num_kv, hdim = (
                    config["num_attention_heads"],
                    config["num_kv_heads"],
                    config["head_dim"],
                )

                torch.manual_seed(config["seed"])
                seq_lens = torch.randint(
                    max_seq_len // 2, max_seq_len + 1, (bs,), device=config["device"]
                )
                q = torch.randn(
                    bs, num_q * hdim, dtype=config["dtype"], device=config["device"]
                )
                k = torch.randn(
                    bs, num_kv * hdim, dtype=config["dtype"], device=config["device"]
                )
                v = torch.randn(
                    bs, num_kv * hdim, dtype=config["dtype"], device=config["device"]
                )

                def run(enable_rope_fusion):
                    backend, model_runner = self._build_trtllm_backend(
                        config, enable_rope_fusion=enable_rope_fusion
                    )
                    layer = _create_layer(config)
                    rotary = _create_rotary_emb(config)
                    _populate_kv_cache(bs, seq_lens, model_runner, layer, config)
                    forward_batch = _create_decode_forward_batch(
                        bs, seq_lens, backend, model_runner, config
                    )
                    backend.init_forward_metadata(forward_batch)
                    if enable_rope_fusion:
                        return backend.forward_decode(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            layer,
                            forward_batch,
                            cos_sin_cache=rotary.cos_sin_cache,
                            is_neox_style=rotary.is_neox_style,
                        )
                    else:
                        q_rope, k_rope = rotary(
                            forward_batch.positions, q.clone(), k.clone()
                        )
                        return backend.forward_decode(
                            q_rope, k_rope, v.clone(), layer, forward_batch
                        )

                out_fused = run(enable_rope_fusion=True)
                out_unfused = run(enable_rope_fusion=False)
                _compare_outputs(
                    self,
                    out_fused,
                    out_unfused,
                    rtol=config["rtol"],
                    atol=config["atol"],
                    label=f"[rope_fusion_decode/{config['name']}]",
                )

    def test_rope_fusion_extend_output_match(self):
        """Fused vs unfused extend should produce the same attention output."""
        for tc in TEST_CASES["rope_fusion_extend"]:
            with self.subTest(name=tc["name"]):
                config = self._merge_config(tc)
                seq_lens_list = config["seq_lens_list"]
                bs = len(seq_lens_list)
                total_num_tokens = sum(seq_lens_list)
                num_q, num_kv, hdim = (
                    config["num_attention_heads"],
                    config["num_kv_heads"],
                    config["head_dim"],
                )

                torch.manual_seed(config["seed"])
                seq_lens = torch.tensor(
                    seq_lens_list, dtype=torch.int32, device=config["device"]
                )
                q = torch.randn(
                    total_num_tokens,
                    num_q * hdim,
                    dtype=config["dtype"],
                    device=config["device"],
                )
                k = torch.randn(
                    total_num_tokens,
                    num_kv * hdim,
                    dtype=config["dtype"],
                    device=config["device"],
                )
                v = torch.randn(
                    total_num_tokens,
                    num_kv * hdim,
                    dtype=config["dtype"],
                    device=config["device"],
                )

                def run(enable_rope_fusion):
                    backend, model_runner = self._build_trtllm_backend(
                        config, enable_rope_fusion=enable_rope_fusion
                    )
                    layer = _create_layer(config)
                    rotary = _create_rotary_emb(config)
                    forward_batch = _create_extend_forward_batch(
                        bs, seq_lens, backend, model_runner, config
                    )
                    backend.init_forward_metadata(forward_batch)
                    if enable_rope_fusion:
                        return backend.forward_extend(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            layer,
                            forward_batch,
                            cos_sin_cache=rotary.cos_sin_cache,
                            is_neox_style=rotary.is_neox_style,
                        )
                    else:
                        q_rope, k_rope = rotary(
                            forward_batch.positions, q.clone(), k.clone()
                        )
                        return backend.forward_extend(
                            q_rope, k_rope, v.clone(), layer, forward_batch
                        )

                out_fused = run(enable_rope_fusion=True)
                out_unfused = run(enable_rope_fusion=False)
                _compare_outputs(
                    self,
                    out_fused,
                    out_unfused,
                    rtol=config["rtol"],
                    atol=config["atol"],
                    label=f"[rope_fusion_extend/{config['name']}]",
                )


if __name__ == "__main__":
    unittest.main()
