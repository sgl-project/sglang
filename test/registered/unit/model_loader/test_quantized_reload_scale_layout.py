"""Quantized reload scales follow their destination projection's layout."""

import math
import socket
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen3Config

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_loader.loader import DefaultModelLoader, QuantizedRLModelLoader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def values(shape, version=0):
    base = (
        torch.arange(math.prod(shape), device="cuda").reshape(shape) % 31 - 15
    ).float() / 64
    if len(shape) == 2:
        base *= ((torch.arange(shape[0], device="cuda") + version) % 17 + 1)[:, None]
        base *= (torch.arange(shape[1], device="cuda") // max(shape[1] // 4, 1) + 1)[
            None, :
        ]
    return (base + version / 128).to(torch.bfloat16)


def checkpoint(version=0, kind="qwen2", kv_heads=8, num_layers=1):
    weights = []
    for name, shape in {
        "model.embed_tokens.weight": (512, 512),
        "lm_head.weight": (512, 512),
        "model.norm.weight": (512,),
        "model.layers.0.input_layernorm.weight": (512,),
        "model.layers.0.post_attention_layernorm.weight": (512,),
        **{
            f"model.layers.0.self_attn.{p}.weight": (
                kv_heads * 64 if p in ("k_proj", "v_proj") else 512,
                512,
            )
            for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        },
        **{
            f"model.layers.0.self_attn.{p}.bias": (
                kv_heads * 64 if p in ("k_proj", "v_proj") else 512,
            )
            for p in ("q_proj", "k_proj", "v_proj")
            if kind == "qwen2"
        },
        **{
            f"model.layers.0.self_attn.{p}.weight": (64,)
            for p in ("q_norm", "k_norm")
            if kind == "qwen3"
        },
        **{
            f"model.layers.0.mlp.{p}.weight": (1024, 512)
            for p in ("gate_proj", "up_proj")
        },
        "model.layers.0.mlp.down_proj.weight": (512, 1024),
    }.items():
        weights.append((name, values(shape, version)))
    first_layer = [(n, t) for n, t in weights if n.startswith("model.layers.0.")]
    for layer in range(1, num_layers):
        weights.extend(
            (n.replace("model.layers.0.", f"model.layers.{layer}."), t)
            for n, t in first_layer
        )
    return weights


def build_model(
    kind="qwen2", *, kv_heads=8, version=0, weights=None, per_tensor=False, num_layers=1
):
    config_class, model_class = {
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[kind]
    config = config_class(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=kv_heads,
        head_dim=64,
        vocab_size=512,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.device("cuda"):
        model = model_class(config, Fp8Config(is_checkpoint_fp8_serialized=False))
    if per_tensor:
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if hasattr(method, "cutlass_fp8_supported"):
                method.cutlass_fp8_supported = False
                method.use_marlin = False
    loader = QuantizedRLModelLoader(LoadConfig(load_format=LoadFormat.FLASH_RL))
    loader.load_weights_and_postprocess(
        model,
        checkpoint(version, kind, kv_heads, num_layers) if weights is None else weights,
        torch.device("cuda"),
    )
    return model


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return get_parallel().override(
        tp_size=4,
        tp_rank=3,
        tp_group=None,
        attn_tp_size=4,
        attn_tp_rank=3,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=4,
        moe_tp_rank=3,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def assert_same_parameters(actual, expected):
    expected_params = dict(expected.named_parameters())
    for name, param in actual.named_parameters():
        reference = expected_params[name]
        assert param.shape == reference.shape, name
        assert param.stride() == reference.stride(), name
        assert param.dtype == reference.dtype, name
        torch.testing.assert_close(
            param.contiguous().reshape(-1).view(torch.uint8),
            reference.contiguous().reshape(-1).view(torch.uint8),
            rtol=0,
            atol=0,
            msg=name,
        )


def check_reload(model, version=13, changed=False, per_tensor=False):
    kind, kv_heads = model.config.model_type, model.config.num_key_value_heads
    source = checkpoint(version, kind, kv_heads)
    pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
    DefaultModelLoader.restore_weights_before_loading(model, torch.device("cuda"))
    with loading_scope(changed):
        model.load_weights(source)
    DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
    for name, param in model.named_parameters():
        assert param.data_ptr() == pointers[name], name
    reference = build_model(
        kind, kv_heads=kv_heads, version=version, per_tensor=per_tensor
    )
    assert_same_parameters(model, reference)
    return model


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestQuantizedReloadScaleLayout(CustomTestCase):
    def setUp(self):
        if torch.distributed.is_initialized():
            self.skipTest("requires an isolated distributed test process")
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)
        server = ServerArgs(model_path="dummy", device="cuda", tp_size=1)
        publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        self.addCleanup(destroy_distributed_environment)
        initialize_model_parallel()
        self.addCleanup(destroy_model_parallel)
        initialize_dp_attention(server)

    def test_native_model_reload_after_scope_exit(self):
        model = build_model()
        check_reload(model, changed=True)
        check_reload(model, version=29, changed=True)

    def test_native_attention_group_projection_reload_after_scope_exit(self):
        model = build_model("qwen3")
        check_reload(model, changed=True)
        check_reload(model, version=29, changed=True)

    def test_unequal_qkv_and_replicated_kv_match_cold_load(self):
        for kv_heads in (1, 2, 4):
            with self.subTest(kv_heads=kv_heads):
                check_reload(build_model("qwen3", kv_heads=kv_heads), changed=True)

    def test_partial_qkv_and_mlp_updates_preserve_other_rows(self):
        for kind in ("qwen2", "qwen3"):
            for shard in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                with self.subTest(kind=kind, shard=shard):
                    model = build_model(kind, kv_heads=1)
                    before = checkpoint(kind=kind, kv_heads=1)
                    update = {
                        n: t for n, t in checkpoint(13, kind, 1) if f".{shard}." in n
                    }
                    pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
                    with loading_scope(True):
                        model.load_weights(list(update.items()))
                    expected = [(n, update.get(n, t)) for n, t in before]
                    reference = build_model(kind, kv_heads=1, weights=expected)
                    assert_same_parameters(model, reference)
                    for name, param in model.named_parameters():
                        self.assertEqual(param.data_ptr(), pointers[name], name)

    def test_per_tensor_quantization_matches_native_cold_load(self):
        model = build_model("qwen3", kv_heads=1, per_tensor=True)
        check_reload(model, version=0, changed=True, per_tensor=True)
        check_reload(model, version=13, changed=True, per_tensor=True)

    def test_partial_per_tensor_update_rejected_before_any_parameter_changes(self):
        for shard in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            with self.subTest(shard=shard):
                model = build_model("qwen3", kv_heads=1, per_tensor=True)
                pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
                source = checkpoint(13, "qwen3", 1)
                update = [
                    (n, t)
                    for n, t in source
                    if n == "model.norm.weight" or f".{shard}." in n
                ]
                with self.assertRaisesRegex(
                    ValueError, "Partial per-tensor FP8 update"
                ):
                    model.load_weights(update)
                assert_same_parameters(
                    model, build_model("qwen3", kv_heads=1, per_tensor=True)
                )
                for name, param in model.named_parameters():
                    self.assertEqual(param.data_ptr(), pointers[name], name)

    def test_per_tensor_session_accepts_complete_destinations(self):
        model = build_model("qwen3", kv_heads=1, per_tensor=True)
        pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
        DefaultModelLoader.restore_weights_before_loading(model, torch.device("cuda"))
        grouped = {}
        for name, tensor in checkpoint(13, "qwen3", 1):
            target, _, _ = QuantizedRLModelLoader._resolve_stacked_info(name)
            grouped.setdefault(target, []).append((name, tensor))
        for source in reversed(list(grouped.values())):
            with loading_scope(True):
                model.load_weights(source)
            self.assertFalse(hasattr(model, "_quantized_rl_pending"))
        DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
        assert_same_parameters(
            model, build_model("qwen3", kv_heads=1, version=13, per_tensor=True)
        )
        for name, param in model.named_parameters():
            self.assertEqual(param.data_ptr(), pointers[name], name)

    def test_per_tensor_session_rejects_partial_chunks_without_host_cache(self):
        model = build_model("qwen3", kv_heads=1, per_tensor=True)
        original = build_model("qwen3", kv_heads=1, per_tensor=True)
        DefaultModelLoader.restore_weights_before_loading(model, torch.device("cuda"))
        for shard in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            update = [
                (n, t)
                for n, t in checkpoint(13, "qwen3", 1)
                if n == "model.norm.weight" or f".{shard}." in n
            ]
            with self.assertRaisesRegex(ValueError, "Partial per-tensor FP8 update"):
                model.load_weights(update)
            assert_same_parameters(model, original)
            self.assertFalse(hasattr(model, "_quantized_rl_pending"))
        model.load_weights(checkpoint(13, "qwen3", 1))
        DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
        assert_same_parameters(
            model, build_model("qwen3", kv_heads=1, version=13, per_tensor=True)
        )

    def test_interleaved_layers_cannot_accumulate_pending_bf16_matrices(self):
        model = build_model("qwen3", kv_heads=1, per_tensor=True, num_layers=16)
        snapshot = {
            n: p.contiguous().reshape(-1).view(torch.uint8).clone()
            for n, p in model.named_parameters()
        }
        pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
        DefaultModelLoader.restore_weights_before_loading(model, torch.device("cuda"))
        source = [
            (n, t)
            for n, t in checkpoint(13, "qwen3", 1, 16)
            if ".q_proj." in n or n == "model.norm.weight"
        ]
        with self.assertRaisesRegex(ValueError, "Partial per-tensor FP8 update"):
            model.load_weights(source)
        self.assertFalse(hasattr(model, "_quantized_rl_pending"))
        for name, param in model.named_parameters():
            self.assertEqual(param.data_ptr(), pointers[name], name)
            torch.testing.assert_close(
                param.contiguous().reshape(-1).view(torch.uint8),
                snapshot[name],
                rtol=0,
                atol=0,
            )
        DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))

    def test_noncanonical_reload_names_are_rejected_before_mutation(self):
        model = build_model("qwen3", kv_heads=1)
        sources = dict(checkpoint(13, "qwen3", 1))
        snapshot = {
            n: (p.data_ptr(), p.contiguous().view(torch.uint8).clone())
            for n, p in model.named_parameters()
        }
        for canonical in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
        ):
            with self.subTest(name=canonical):
                update = [
                    ("model.norm.weight", sources["model.norm.weight"]),
                    (canonical.removeprefix("model."), sources[canonical]),
                ]
                with self.assertRaisesRegex(ValueError, "canonical checkpoint names"):
                    model.load_weights(update)
                for name, param in model.named_parameters():
                    pointer, data = snapshot[name]
                    self.assertEqual(param.data_ptr(), pointer)
                    torch.testing.assert_close(
                        param.contiguous().view(torch.uint8), data, rtol=0, atol=0
                    )

    def test_qwen2_walker_reload_matches_cold_load(self):
        from sglang.srt.environ import envs

        with patch.object(
            envs.SGLANG_ENABLE_WEIGHT_LOADER_V2, "get", return_value=True
        ):
            for per_tensor in (False, True):
                with self.subTest(per_tensor=per_tensor):
                    check_reload(
                        build_model("qwen2", kv_heads=1, per_tensor=per_tensor),
                        changed=True,
                        per_tensor=per_tensor,
                    )

    def test_qwen2_walker_tied_head_post_copy_occurs_once(self):
        from sglang.srt.environ import envs

        model = build_model("qwen2", kv_heads=1)
        model.config.tie_word_embeddings = True
        head = model.lm_head.weight
        original = head.weight_loader
        copies = []

        def load_head(*args, **kwargs):
            copies.append(True)
            return original(*args, **kwargs)

        head.weight_loader = load_head
        source = checkpoint(13, "qwen2", 1)
        embedding = dict(source)["model.embed_tokens.weight"]
        with patch.object(
            envs.SGLANG_ENABLE_WEIGHT_LOADER_V2, "get", return_value=True
        ):
            model.load_weights(source)
        self.assertEqual(len(copies), 1)
        torch.testing.assert_close(head, embedding, rtol=0, atol=0)
        expected = [(n, embedding if n == "lm_head.weight" else t) for n, t in source]
        assert_same_parameters(
            model, build_model("qwen2", kv_heads=1, weights=expected)
        )

    def test_unsupported_native_loaders_fail_before_initialization(self):
        from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
        from sglang.srt.models.whisper import WhisperForConditionalGeneration

        for native in (
            WhisperForConditionalGeneration.load_weights,
            MiMoV2ForCausalLM.load_weights,
        ):
            with self.subTest(native=native.__qualname__):
                model = torch.nn.Linear(4, 4, device="cuda")
                before = model.weight.clone()
                model.load_weights = native.__get__(model)
                original = model.load_weights
                loader = QuantizedRLModelLoader(
                    LoadConfig(load_format=LoadFormat.FLASH_RL)
                )
                with self.assertRaisesRegex(ValueError, "deferred FP8-write contract"):
                    loader.load_weights_and_postprocess(model, [], torch.device("cuda"))
                self.assertIs(model.load_weights, original)
                self.assertFalse(hasattr(model, "original_weights_rebuild_keys"))
                torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def test_failed_native_load_restores_fp8_storage(self):
        model = build_model("qwen3", kv_heads=1)
        pointers = {n: p.data_ptr() for n, p in model.named_parameters()}

        def fail(weights):
            Qwen3ForCausalLM.load_weights(model, weights)
            self.assertEqual(
                model.model.layers[0].self_attn.qkv_proj.weight.dtype,
                torch.float8_e4m3fn,
            )
            raise RuntimeError("native loader failed")

        with self.assertRaisesRegex(RuntimeError, "native loader failed"):
            QuantizedRLModelLoader.rebinding_and_load_weights(
                model,
                fail,
                [
                    (n, t)
                    for n, t in checkpoint(13, "qwen3", 1)
                    if any(f".{p}." in n for p in ("q_proj", "k_proj", "v_proj"))
                ],
            )
        assert_same_parameters(model, build_model("qwen3", kv_heads=1))
        for name, param in model.named_parameters():
            self.assertEqual(param.data_ptr(), pointers[name], name)

    def test_failed_deferred_parameter_write_restores_storage_and_loader(self):
        model = build_model("qwen3", kv_heads=1)
        param = model.model.layers[0].self_attn.qkv_proj.weight
        pointer = param.data_ptr()
        snapshot = param.contiguous().view(torch.uint8).clone()

        def fail(target, weight, shard):
            self.assertEqual(target.dtype, torch.bfloat16)
            raise RuntimeError("deferred write failed")

        param.weight_loader = fail
        source = [
            (n, t)
            for n, t in checkpoint(13, "qwen3", 1)
            if any(f".{shard}_proj." in n for shard in ("q", "k", "v"))
        ]
        with self.assertRaisesRegex(RuntimeError, "deferred write failed"):
            model.load_weights(source)
        self.assertEqual(param.data_ptr(), pointer)
        self.assertIs(param.weight_loader, fail)
        torch.testing.assert_close(
            param.contiguous().view(torch.uint8), snapshot, rtol=0, atol=0
        )

    def test_fnuz_per_tensor_reload_matches_cold_load(self):
        from sglang.srt.layers.quantization import fp8
        from sglang.srt.layers.quantization.fp8_utils import input_to_float8

        with patch.object(
            fp8,
            "input_to_float8",
            lambda x: input_to_float8(x, dtype=torch.float8_e4m3fnuz),
        ):
            model = build_model("qwen3", kv_heads=1, per_tensor=True)
            check_reload(model, version=13, per_tensor=True)

    def test_per_tensor_output_buffer_matches_reference_without_weight_temporary(self):
        from sglang.srt.layers.quantization.fp8_utils import input_to_float8

        x = values((2048, 2048), 13)
        for dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            with self.subTest(dtype=dtype):
                expected, expected_scale = input_to_float8(x, dtype=dtype)
                output = torch.empty_like(x, dtype=dtype)
                input_to_float8(x, dtype=dtype, out=output)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                actual, scale = input_to_float8(x, dtype=dtype, out=output)
                torch.cuda.synchronize()
                extra = torch.cuda.max_memory_allocated() - baseline
                self.assertLess(extra, 64 * 1024)
                self.assertEqual(actual.data_ptr(), output.data_ptr())
                torch.testing.assert_close(
                    actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
                )
                torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)

    def test_empty_session_keeps_native_layout_and_memory(self):
        model = build_model("qwen3", kv_heads=1)
        pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
        DefaultModelLoader.restore_weights_before_loading(model, torch.device("cuda"))
        model.load_weights([])
        DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
        assert_same_parameters(model, build_model("qwen3", kv_heads=1))
        for name, param in model.named_parameters():
            self.assertEqual(param.data_ptr(), pointers[name], name)


if __name__ == "__main__":
    unittest.main()
