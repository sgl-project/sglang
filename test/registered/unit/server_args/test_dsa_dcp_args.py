"""RoPE DSA DCP launch policy, before model loading or CUDA graph capture."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import (
    _dsa_kv_cache_dtype_default,
    _dsa_split_backend_resolution,
    collect_model_override_declarations,
    declare_resolution,
    resolution_result,
    run_post_process_pass,
    validate_declarations,
)
from sglang.srt.arg_groups.parallel_hook import handle_deprecated_dp_attention
from sglang.srt.environ import EnvField, envs
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDsaDcpArgs(CustomTestCase):
    def test_full_resolution_preserves_dcp_backends_and_rejects_incompatible_decode(
        self,
    ):
        """Exercise handler ordering with a real local config, without weights."""
        config = dict(
            architectures=["GlmMoeDsaForCausalLM"],
            model_type="glm_moe_dsa",
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_attention_heads=64,
            num_key_value_heads=64,
            num_hidden_layers=2,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            vocab_size=128,
            max_position_embeddings=2048,
            kv_lora_rank=8,
            q_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            topk_method="greedy",
            scoring_func="softmax",
            index_topk=4,
            index_head_dim=8,
            index_n_heads=2,
        )
        # Resolution can write both environment strings and EnvField's None flag.
        flags = [
            (field, field._set_to_none)
            for cls in type(envs).__mro__
            for field in vars(cls).values()
            if isinstance(field, EnvField)
        ]
        try:
            with (
                tempfile.TemporaryDirectory() as model_dir,
                patch.dict(os.environ, dict(os.environ)),
                patch(
                    "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
                    return_value=275 * 1024,
                ),
            ):
                Path(model_dir, "config.json").write_text(json.dumps(config))
                common = dict(
                    model_path=model_dir,
                    device="cuda",
                    tp_size=4,
                    dcp_size=2,
                    random_seed=0,
                )
                for kv_dtype, expected in (
                    ("bfloat16", "bfloat16"),
                    ("auto", "fp8_e4m3"),
                ):
                    with self.subTest(kv_dtype=kv_dtype):
                        args = ServerArgs(**common, kv_cache_dtype=kv_dtype)
                        args.resolve_once()
                        self.assertEqual(
                            resolution_result(args, "attention_backend"), "dsa"
                        )
                        self.assertEqual(
                            resolution_result(args, "dsa_prefill_backend"), "trtllm"
                        )
                        self.assertEqual(
                            resolution_result(args, "dsa_decode_backend"), "trtllm"
                        )
                        self.assertEqual(
                            resolution_result(args, "kv_cache_dtype"), expected
                        )
                        self.assertEqual(resolution_result(args, "page_size"), 64)
                        self.assertIsNone(args.dsa_prefill_backend)
                        self.assertIsNone(args.dsa_decode_backend)
                with self.assertRaisesRegex(
                    ValueError, "requires trtllm.*flashmla_sparse"
                ):
                    ServerArgs(
                        **common, dsa_decode_backend="flashmla_sparse"
                    ).resolve_once()
        finally:
            for field, old_flag in flags:
                field._set_to_none = old_flag

    def setUp(self):
        self.platform = override_platform(
            is_cuda=True, is_hip=False, is_npu=False, is_xpu=False, is_sm100=True
        )
        self.platform.__enter__()
        self.addCleanup(self.platform.__exit__, None, None, None)
        capability = patch("torch.cuda.get_device_capability", return_value=(10, 3))
        self.capability = capability.start()
        self.addCleanup(capability.stop)

    @staticmethod
    def _args(arch="GlmMoeDsaForCausalLM", **kwargs):
        args = ServerArgs(
            **{"model_path": "dummy", "tp_size": 4, "dcp_size": 2, **kwargs}
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch], index_topk=2048)
        )
        return args

    @staticmethod
    def _resolve(args):
        handle_deprecated_dp_attention(args)
        hf_config = args._model_config.hf_config
        declarations = collect_model_override_declarations(
            hf_config.architectures[0], args, hf_config
        )
        validate_declarations(args, declarations)
        for source, fields in declarations:
            declare_resolution(args, source, **fields)

    def test_unsupported_combinations_rejected_before_runtime(self):
        """Unsupported cache/spec paths must fail before graph warmup or KV writes."""
        cases = (
            ({"enable_hisparse": True}, "--enable-hisparse"),
            ({"enable_prefill_cp": True}, "prefill context parallelism"),
            ({"attn_cp_size": 2}, "prefill context parallelism"),
            ({"enable_hierarchical_cache": True}, "HiCache"),
            ({"hicache_storage_backend": "file"}, "HiCache"),
            ({"enable_unified_cache_external_linker": True}, "HiCache"),
            ({"enable_unified_memory": True}, "--enable-unified-memory"),
            ({"disaggregation_mode": "prefill"}, "PD disaggregation"),
            ({"disaggregation_mode": "decode"}, "PD disaggregation"),
            ({"speculative_algorithm": "EAGLE3"}, "speculative decoding"),
            ({"speculative_algorithm": "DFLASH"}, "speculative decoding"),
            ({"dcp_replicate_q_proj": True}, "--dcp-replicate-q-proj"),
            ({"kv_cache_dtype": "fp8_e5m2"}, "KV cache"),
        )
        for kwargs, message in cases:
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                self._resolve(self._args(**kwargs))

    def test_chain_eagle_supported_and_other_shapes_rejected(self):
        supported = dict(
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=6,
        )
        for dcp_size in (2, 4):
            self._resolve(self._args(**supported, dcp_size=dcp_size))
        for changed in (
            {"speculative_eagle_topk": 2},
            {"speculative_eagle_topk": None},
            {"speculative_num_steps": None},
            {"speculative_num_steps": 0},
            {"speculative_num_draft_tokens": 7},
            {"speculative_adaptive": True},
            {"enable_multi_layer_eagle": True},
            {"speculative_draft_model_path": "different-draft"},
            {"speculative_draft_attention_backend": "flashinfer"},
            {"speculative_draft_kv_cache_dtype": "bfloat16"},
        ):
            with (
                self.subTest(**changed),
                self.assertRaisesRegex(ValueError, "RoPE DSA DCP.*EAGLE"),
            ):
                self._resolve(self._args(**(supported | changed)))

    def test_non_trtllm_backends_rejected_for_each_phase(self):
        for field in ("dsa_prefill_backend", "dsa_decode_backend"):
            for backend in ("flashmla_sparse", "flashmla_kv", "tilelang", "fa3"):
                with self.subTest(field=field, backend=backend):
                    with self.assertRaisesRegex(ValueError, "requires.*trtllm"):
                        self._resolve(self._args(**{field: backend}))
        for field in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "requires.*dsa"),
            ):
                self._resolve(self._args(**{field: "flashinfer"}))

    def test_hardware_scope(self):
        for capability in ((9, 0), (11, 0), (12, 0), (12, 1)):
            with (
                self.subTest(capability=capability),
                self.assertRaisesRegex(ValueError, "SM100.*SM103"),
            ):
                self.capability.return_value = capability
                self._resolve(self._args())

    def test_attention_dp_group_contains_dcp_group(self):
        for kwargs in (
            {"attn_dp_size": 4},
            {"dp_size": 4, "enable_dp_attention": True},
            {"dcp_size": 3},
            {"dcp_size": 8},
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "attention TP size.*divisible"),
            ):
                self._resolve(self._args(**kwargs))

    def test_supported_defaults_preserve_raw_inputs(self):
        """BF16 DCP must not inherit the ordinary flashmla_sparse prefill default."""
        for capability in ((10, 0), (10, 3)):
            self.capability.return_value = capability
            for kv_dtype in ("auto", "fp8_e4m3", "bf16", "bfloat16"):
                for dcp_size in (2, 4):
                    with self.subTest(
                        capability=capability, kv_dtype=kv_dtype, dcp_size=dcp_size
                    ):
                        args = self._args(kv_cache_dtype=kv_dtype, dcp_size=dcp_size)
                        self._resolve(args)
                        run_post_process_pass(args, _dsa_kv_cache_dtype_default)
                        run_post_process_pass(args, _dsa_split_backend_resolution)
                        self.assertEqual(
                            resolution_result(args, "dsa_prefill_backend"), "trtllm"
                        )
                        self.assertEqual(
                            resolution_result(args, "dsa_decode_backend"), "trtllm"
                        )
                        self.assertIsNone(args.dsa_prefill_backend)
                        self.assertIsNone(args.dsa_decode_backend)
                        self.assertEqual(args.kv_cache_dtype, kv_dtype)

    def test_supported_parallel_composition(self):
        for kwargs in (
            {"arch": "DeepseekV32ForCausalLM"},
            {"arch": "DeepseekV3ForCausalLM"},
            {"ep_size": 4},
            {"attn_dp_size": 2},
            {"dp_size": 2, "enable_dp_attention": True},
            {"quantization": "modelopt_fp4"},
            {"dcp_comm_backend": "ag_rs", "enable_symm_mem": True},
            {"dcp_comm_backend": "a2a"},
            {"dcp_comm_backend": "fi_a2a"},
        ):
            with self.subTest(**kwargs):
                args = self._args(**kwargs)
                self._resolve(args)
                self.assertEqual(resolution_result(args, "attention_backend"), "dsa")

    def test_dsa_dcp_policy_does_not_affect_other_paths(self):
        for kwargs in ({"dcp_size": 1}, {"arch": "Glm5NextForConditionalGeneration"}):
            with self.subTest(**kwargs):
                args = self._args(speculative_algorithm="EAGLE", **kwargs)
                self._resolve(args)
                self.assertIsNone(resolution_result(args, "dsa_prefill_backend"))
        args = self._args(arch="DeepseekV3ForCausalLM", speculative_algorithm="EAGLE")
        args._model_config.hf_config.index_topk = None
        self._resolve(args)
        self.assertEqual(resolution_result(args, "attention_backend"), "trtllm_mla")
        for platform in ("is_hip", "is_npu"):
            with (
                self.subTest(platform=platform),
                override_platform(is_cuda=False, **{platform: True}),
            ):
                args = self._args(speculative_algorithm="EAGLE")
                self._resolve(args)
                self.assertIsNone(resolution_result(args, "dsa_prefill_backend"))


if __name__ == "__main__":
    unittest.main()
