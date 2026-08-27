"""CPU tests for residue NVFP4 config selection and stock-path guards.

These run without a GPU: they cover the checkpoint-driven wrap
(residue_kernel_metadata.json next to the weights selects the residue
config), the explicit --quantization modelopt_fp4_residue contract, and
that layers without residue metadata resolve to stock-behaving methods.
"""

import json
from types import SimpleNamespace

import pytest

from sglang.srt.layers.quantization import (
    BASE_QUANTIZATION_METHODS,
    get_quantization_config,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptFp8LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.residue_nvfp4 import (
    METADATA_FILENAME,
    ResidueMetadataError,
    ResidueMode,
)
from sglang.srt.layers.quantization.residue_nvfp4.config import (
    ModelOptFp4ResidueConfig,
    ModelOptMixedPrecisionResidueConfig,
    maybe_wrap_residue_fp4_config,
)
from sglang.srt.layers.quantization.residue_nvfp4.linear_method import (
    ModelOptFp4ResidueLinearMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

K = 4096


def make_fp4_config() -> ModelOptFp4Config:
    return ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo="auto",
        group_size=16,
        exclude_modules=["lm_head"],
        packed_modules_mapping={},
    )


def make_mixed_config() -> ModelOptMixedPrecisionConfig:
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.layers.0.self_attn.k_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.layers.0.self_attn.v_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.layers.0.fp8_proj": {"quant_algo": "FP8"},
            },
            "packed_modules_mapping": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
        }
    )


def write_metadata(model_dir, layers=None):
    layers = layers or {
        "model.layers.0.self_attn.q_proj": {
            "salient_indices": list(range(K)),
            "num_salient": K,
            "runtime_mode": "mext_r1",
        }
    }
    (model_dir / METADATA_FILENAME).write_text(
        json.dumps(
            {"layers": layers, "global": {"residue_ratio": 1.0, "block_size": 8}}
        )
    )


def model_config_for(path, quantization="modelopt_fp4"):
    return SimpleNamespace(model_path=str(path), quantization=quantization)


class TestRegistry:
    def test_residue_method_registered(self):
        assert (
            BASE_QUANTIZATION_METHODS["modelopt_fp4_residue"]
            is ModelOptFp4ResidueConfig
        )
        assert (
            get_quantization_config("modelopt_fp4_residue") is ModelOptFp4ResidueConfig
        )

    def test_from_config_alone_refuses(self):
        with pytest.raises(ResidueMetadataError, match="metadata"):
            ModelOptFp4ResidueConfig.from_config({"quant_algo": "NVFP4"})


class TestCheckpointDrivenWrap:
    def test_metadata_present_wraps(self, tmp_path):
        write_metadata(tmp_path)
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path), make_fp4_config()
        )
        assert isinstance(cfg, ModelOptFp4ResidueConfig)
        spec = cfg.residue_spec.spec_for("model.layers.0.self_attn.q_proj")
        assert spec.mode is ResidueMode.MEXT_R1
        # Parent fields survive the wrap.
        assert cfg.group_size == 16
        assert cfg.exclude_modules == ["lm_head"]

    def test_metadata_absent_returns_stock_config(self, tmp_path):
        base = make_fp4_config()
        cfg = maybe_wrap_residue_fp4_config(model_config_for(tmp_path), base)
        assert cfg is base
        assert not isinstance(cfg, ModelOptFp4ResidueConfig)

    def test_broken_metadata_raises_never_falls_back(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text("{not json")
        with pytest.raises(ResidueMetadataError):
            maybe_wrap_residue_fp4_config(model_config_for(tmp_path), make_fp4_config())

    def test_explicit_name_requires_metadata(self, tmp_path):
        with pytest.raises(ResidueMetadataError, match="no residue"):
            maybe_wrap_residue_fp4_config(
                model_config_for(tmp_path, "modelopt_fp4_residue"),
                make_fp4_config(),
            )

    def test_explicit_name_with_metadata_wraps(self, tmp_path):
        write_metadata(tmp_path)
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path, "modelopt_fp4_residue"), make_fp4_config()
        )
        assert isinstance(cfg, ModelOptFp4ResidueConfig)

    def test_non_fp4_config_passes_through(self, tmp_path):
        write_metadata(tmp_path)
        other = object()
        assert maybe_wrap_residue_fp4_config(model_config_for(tmp_path), other) is other

    def test_awq_checkpoint_rejected(self, tmp_path):
        write_metadata(tmp_path)
        base = make_fp4_config()
        base.is_awq = True
        with pytest.raises(ResidueMetadataError, match="AWQ"):
            maybe_wrap_residue_fp4_config(model_config_for(tmp_path), base)


class TestMixedPrecisionCheckpointDrivenWrap:
    def test_metadata_present_wraps_outer_mixed_config(self, tmp_path):
        write_metadata(tmp_path)
        base = make_mixed_config()
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path, "modelopt_mixed"), base
        )

        assert isinstance(cfg, ModelOptMixedPrecisionResidueConfig)
        assert isinstance(cfg.nvfp4_config, ModelOptFp4ResidueConfig)
        assert cfg.fp8_config is base.fp8_config
        assert cfg.nvfp4a16_config is base.nvfp4a16_config
        assert cfg.mtp_quant_config is base.mtp_quant_config

    def test_metadata_absent_returns_stock_mixed_config(self, tmp_path):
        base = make_mixed_config()
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path, "modelopt_mixed"), base
        )
        assert cfg is base

    def test_only_dense_nvfp4_is_intercepted(self, tmp_path):
        from sglang.srt.layers.linear import ReplicatedLinear

        write_metadata(tmp_path)
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path, "modelopt_mixed"), make_mixed_config()
        )
        layer = ReplicatedLinear.__new__(ReplicatedLinear)

        qkv_method = cfg.get_quant_method(layer, "model.layers.0.self_attn.qkv_proj")
        assert isinstance(qkv_method, ModelOptFp4ResidueLinearMethod)
        assert qkv_method.layer_spec is not None
        assert qkv_method.layer_spec.mode is ResidueMode.MEXT_R1

        fp8_method = cfg.get_quant_method(layer, "model.layers.0.fp8_proj")
        assert type(fp8_method) is ModelOptFp8LinearMethod

    def test_metadata_less_nvfp4_dense_layer_keeps_exact_stock_method(self, tmp_path):
        from sglang.srt.layers.linear import ReplicatedLinear

        write_metadata(tmp_path)
        base = make_mixed_config()
        base.quantized_layers["model.layers.1.down_proj"] = {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path, "modelopt_mixed"), base
        )
        layer = ReplicatedLinear.__new__(ReplicatedLinear)
        method = cfg.get_quant_method(layer, "model.layers.1.down_proj")

        assert type(method) is ModelOptFp4LinearMethod

    def test_non_linear_module_matches_stock_mixed_config(self, tmp_path):
        import torch

        write_metadata(tmp_path)
        base = make_mixed_config()
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path, "modelopt_mixed"), base
        )
        layer = torch.nn.Module()
        prefix = "model.layers.0.self_attn.qkv_proj"
        assert cfg.get_quant_method(layer, prefix) == base.get_quant_method(
            layer, prefix
        )


class TestMethodResolution:
    def _residue_config(self, tmp_path):
        write_metadata(tmp_path)
        return maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path), make_fp4_config()
        )

    def test_residue_layer_gets_residue_method(self, tmp_path):
        cfg = self._residue_config(tmp_path)
        method = ModelOptFp4ResidueLinearMethod(cfg, "model.layers.0.self_attn.q_proj")
        assert method.layer_spec is not None
        assert method.layer_spec.mode is ResidueMode.MEXT_R1
        assert method.layer_spec.k_base == K

    def test_fused_qkv_resolves_through_parts(self, tmp_path):
        cfg = self._residue_config(tmp_path)
        method = ModelOptFp4ResidueLinearMethod(
            cfg, "model.layers.0.self_attn.qkv_proj"
        )
        assert method.layer_spec is not None

    def test_layer_without_metadata_is_stock(self, tmp_path):
        cfg = self._residue_config(tmp_path)
        method = ModelOptFp4ResidueLinearMethod(cfg, "model.layers.0.mlp.down_proj")
        assert method.layer_spec is None
        # Stock-path guard: every stage delegates to the parent class.
        assert isinstance(method, ModelOptFp4LinearMethod)
        assert type(method).process_weights_after_loading.__qualname__.startswith(
            "ModelOptFp4ResidueLinearMethod"
        )

    def test_k_ext_layer_resolves(self):
        n_sal = K // 4
        indices = sorted(i for b in range(0, K, 8) for i in range(b, b + 2))
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            (tmp_path / METADATA_FILENAME).write_text(
                json.dumps(
                    {
                        "layers": {
                            "model.layers.0.mlp.up_proj": {
                                "salient_indices": indices,
                                "num_salient": n_sal,
                                "runtime_mode": "extended_k",
                            }
                        },
                        "global": {"residue_ratio": 0.25, "block_size": 8},
                        "extended_linear_dims": {
                            "model.layers.0.mlp.up_proj": K + n_sal
                        },
                    }
                )
            )
            cfg = maybe_wrap_residue_fp4_config(
                model_config_for(tmp_path), make_fp4_config()
            )
            method = ModelOptFp4ResidueLinearMethod(cfg, "model.layers.0.mlp.up_proj")
            assert method.layer_spec.mode is ResidueMode.K_EXT
            assert method.layer_spec.k_ext == K + n_sal
            assert method.layer_spec.residue_per_block == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


class TestKExtCreateWeights:
    """create_weights must size the k_ext weight to the EXTENDED K (per rank),
    and install the two-range shard plan on row-parallel layers."""

    def _method(self, tmp_path, n_sal=K // 4):
        import torch  # noqa: F401

        indices = sorted(i for b in range(0, K, 8) for i in range(b, b + 2))
        (tmp_path / METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "layers": {
                        "l.down_proj": {
                            "salient_indices": indices,
                            "num_salient": n_sal,
                            "runtime_mode": "extended_k",
                        }
                    },
                    "global": {"residue_ratio": 0.25, "block_size": 8},
                    "extended_linear_dims": {"l.down_proj": K + n_sal},
                }
            )
        )
        cfg = maybe_wrap_residue_fp4_config(
            model_config_for(tmp_path), make_fp4_config()
        )
        return ModelOptFp4ResidueLinearMethod(cfg, "l.down_proj")

    def test_column_parallel_gets_full_extension(self, tmp_path):
        import torch

        method = self._method(tmp_path)
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=K,
            output_partition_sizes=[512],
            input_size=K,
            output_size=512,
            params_dtype=torch.bfloat16,
        )
        n_ext = K + K // 4
        assert layer.weight.shape == (512, n_ext // 2)
        assert layer.weight_scale.shape == (512, n_ext // 16)
        assert getattr(layer, "_residue_shard_plan", None) is None

    def test_row_parallel_gets_sharded_extension_and_plan(self, tmp_path):
        import torch

        def dummy_loader(param, loaded_weight, *args, **kwargs):
            return None

        method = self._method(tmp_path)
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=K // 2,
            output_partition_sizes=[512],
            input_size=K,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=dummy_loader,
        )
        n_ext_shard = (K + K // 4) // 2
        assert layer.weight.shape == (512, n_ext_shard // 2)
        plan = layer._residue_shard_plan
        assert plan is not None and plan.tp_size == 2
        # The loaders were wrapped for the extended-column permutation.
        assert layer.weight.weight_loader is not dummy_loader
        assert layer.weight_scale.weight_loader is not dummy_loader

    def test_standard_layer_weight_shape_unchanged(self, tmp_path):
        import torch

        method = self._method(tmp_path)
        # Different prefix -> no residue spec -> stock contract.
        method_std = ModelOptFp4ResidueLinearMethod(method.quant_config, "l.other_proj")
        layer = torch.nn.Module()
        method_std.create_weights(
            layer,
            input_size_per_partition=K,
            output_partition_sizes=[512],
            input_size=K,
            output_size=512,
            params_dtype=torch.bfloat16,
        )
        assert layer.weight.shape == (512, K // 2)

    def test_pwal_prepares_base_k_fold_layout(self, tmp_path, monkeypatch):
        """K-ext keeps the full scale for prefill and derives a separately
        swizzled base-prefix scale for decode fold before parent PWAL can
        overwrite the raw Parameter storage."""
        import torch

        method = self._method(tmp_path)
        n = 130
        k_ext = K + K // 4
        layer = torch.nn.Module()
        layer.weight = torch.zeros(n, k_ext // 2, dtype=torch.uint8)
        layer.weight_scale = torch.ones(n, k_ext // 16, dtype=torch.float8_e4m3fn)

        full_scale_marker = torch.empty(1, dtype=torch.float8_e4m3fn)

        def fake_parent_pwal(_self, target):
            # Simulate both parent effects relevant to the residue method:
            # aliasing destroys the raw-scale contents and N gets padded.
            target.weight_scale.zero_()
            target.weight = torch.zeros(160, k_ext // 2, dtype=torch.uint8)
            target.weight_scale_interleaved = full_scale_marker
            target.weights_padding_cols = 0

        monkeypatch.setattr(
            ModelOptFp4LinearMethod,
            "process_weights_after_loading",
            fake_parent_pwal,
        )

        seen = {}

        def fake_swizzle(scale):
            seen["raw_base"] = scale.clone()
            return torch.full_like(scale, 2)

        import sglang.srt.layers.quantization.utils as quant_utils

        monkeypatch.setattr(quant_utils, "swizzle_blockscale", fake_swizzle)

        import sglang.srt.layers.quantization.residue_nvfp4.warmup as warmup

        monkeypatch.setattr(
            warmup,
            "register_fold_shape",
            lambda *args, **kwargs: seen.setdefault("shape", (args, kwargs)),
        )

        method.process_weights_after_loading(layer)

        assert seen["raw_base"].shape == (n, K // 16)
        assert torch.all(seen["raw_base"].float() == 1)
        assert layer.weight_scale_interleaved is full_scale_marker
        assert layer.weight_scale_base.shape == (n, K // 16)
        assert layer._mext_decode_k_base == K
        assert layer._residue_fold_eligible is True
        assert seen["shape"] == ((160, K, k_ext), {"is_mext_r1": False})


class TestStockSurfaceGuard:
    """Attention, MoE, embeddings, and excluded layers must resolve exactly as
    the stock modelopt_fp4 config resolves them -- the residue override only
    intercepts LinearBase."""

    def _cfg_pair(self, tmp_path):
        write_metadata(tmp_path)
        base = make_fp4_config()
        wrapped = maybe_wrap_residue_fp4_config(model_config_for(tmp_path), base)
        return base, wrapped

    def test_non_linear_module_defers_to_parent(self, tmp_path):
        import torch

        base, wrapped = self._cfg_pair(tmp_path)
        plain_module = torch.nn.Module()
        assert wrapped.get_quant_method(plain_module, "model.layers.0.foo") == (
            base.get_quant_method(plain_module, "model.layers.0.foo")
        )

    def test_excluded_linear_matches_parent(self, tmp_path):
        from sglang.srt.layers.linear import ReplicatedLinear

        base, wrapped = self._cfg_pair(tmp_path)
        # lm_head is in exclude_modules for both configs.
        layer = ReplicatedLinear.__new__(ReplicatedLinear)
        got = wrapped.get_quant_method(layer, "lm_head")
        want = base.get_quant_method(layer, "lm_head")
        assert type(got) is type(want)

    def test_metadata_less_linear_gets_residue_method_with_stock_behavior(
        self, tmp_path
    ):
        from sglang.srt.layers.linear import ReplicatedLinear

        _, wrapped = self._cfg_pair(tmp_path)
        layer = ReplicatedLinear.__new__(ReplicatedLinear)
        method = wrapped.get_quant_method(layer, "model.layers.5.mlp.down_proj")
        assert isinstance(method, ModelOptFp4ResidueLinearMethod)
        assert method.layer_spec is None  # stock delegation
