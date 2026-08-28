"""CPU tests for the residue NVFP4 metadata contract.

No torch, no GPU: the metadata reader is pure Python by design so these run
anywhere (including the macOS dev host).
"""

import json

import pytest

from sglang.srt.layers.quantization.residue_nvfp4.metadata import (
    METADATA_FILENAME,
    ResidueMetadataError,
    ResidueMode,
    layer_name_candidates,
    load_residue_model_spec,
    parse_residue_metadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

K = 4096  # K_base used throughout; multiple of 16 and of the salient blocks


def k_ext_indices(k_base: int, per_block: int) -> list[int]:
    """Top-`per_block` channels of every 8-channel block, globally sorted --
    the shape the exporter guarantees."""
    out = []
    for block_start in range(0, k_base, 8):
        out.extend(range(block_start, block_start + per_block))
    return sorted(out)


def k_ext_layer(name: str, k_base: int = K, per_block: int = 2) -> tuple[dict, dict]:
    indices = k_ext_indices(k_base, per_block)
    entry = {
        "salient_indices": indices,
        "num_salient": len(indices),
        "runtime_mode": "extended_k",
    }
    return entry, {name: k_base + len(indices)}


def mext_r1_layer(k_base: int = K) -> dict:
    return {
        "salient_indices": list(range(k_base)),
        "num_salient": k_base,
        "runtime_mode": "mext_r1",
    }


def make_metadata(layers: dict, extended_dims: dict | None = None, **extra) -> dict:
    data = {
        "layers": layers,
        "global": {"residue_ratio": 0.25, "block_size": 8},
    }
    if extended_dims:
        data["extended_linear_dims"] = extended_dims
    data.update(extra)
    return data


class TestValidMetadata:
    def test_k_ext_layer_parses(self):
        entry, dims = k_ext_layer("model.layers.0.self_attn.q_proj")
        spec = parse_residue_metadata(
            make_metadata({"model.layers.0.self_attn.q_proj": entry}, dims)
        )
        layer = spec.spec_for("model.layers.0.self_attn.q_proj")
        assert layer is not None
        assert layer.mode is ResidueMode.K_EXT
        assert layer.k_base == K
        assert layer.num_salient == K // 4
        assert layer.k_ext == K + K // 4
        assert layer.ratio == pytest.approx(0.25)
        assert layer.residue_per_block == 2

    @pytest.mark.parametrize("per_block,ratio", [(1, 0.125), (2, 0.25), (4, 0.5)])
    def test_all_supported_k_ext_ratios(self, per_block, ratio):
        entry, dims = k_ext_layer("l", per_block=per_block)
        spec = parse_residue_metadata(make_metadata({"l": entry}, dims))
        layer = spec.spec_for("l")
        assert layer.ratio == pytest.approx(ratio)
        assert layer.residue_per_block == per_block

    def test_mext_r1_layer_parses(self):
        spec = parse_residue_metadata(make_metadata({"l": mext_r1_layer()}))
        layer = spec.spec_for("l")
        assert layer.mode is ResidueMode.MEXT_R1
        assert layer.k_base == K
        assert layer.num_salient == K
        assert layer.k_ext == K  # weight is NOT extended
        assert layer.ratio == pytest.approx(1.0)
        assert layer.salient_indices == ()  # not stored for mext_r1

    def test_unknown_layer_resolves_to_none(self):
        spec = parse_residue_metadata(make_metadata({"l": mext_r1_layer()}))
        assert spec.spec_for("other_layer") is None

    def test_standard_entry_is_no_residue(self):
        spec = parse_residue_metadata(
            make_metadata({"l": {"runtime_mode": "standard"}})
        )
        assert spec.spec_for("l") is None

    def test_moe_all_off_is_accepted(self):
        spec = parse_residue_metadata(
            make_metadata(
                {"l": mext_r1_layer()},
                moe={
                    "granularity": "layer",
                    "layers": {"model.layers.0.mlp": {"impl": "off", "ratio": 0.0}},
                },
            )
        )
        assert spec.spec_for("l") is not None


class TestFusedLayerResolution:
    def test_candidates(self):
        assert layer_name_candidates("model.layers.0.self_attn.qkv_proj") == [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ]
        assert layer_name_candidates("model.layers.0.mlp.gate_up_proj") == [
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
        ]
        assert layer_name_candidates("model.layers.1.linear_attn.in_proj_qkvz") == [
            "model.layers.1.linear_attn.in_proj_qkvz",
            "model.layers.1.linear_attn.in_proj_qkv",
            "model.layers.1.linear_attn.in_proj_z",
        ]

    def test_qkv_resolves_through_parts_when_consistent(self):
        layers, dims = {}, {}
        for part in ("q_proj", "k_proj", "v_proj"):
            entry, d = k_ext_layer(f"m.self_attn.{part}")
            layers[f"m.self_attn.{part}"] = entry
            dims.update(d)
        spec = parse_residue_metadata(make_metadata(layers, dims))
        fused = spec.spec_for("m.self_attn.qkv_proj")
        assert fused is not None
        assert fused.mode is ResidueMode.K_EXT
        assert fused.name == "m.self_attn.qkv_proj"

    def test_conflicting_parts_raise(self):
        e1, d1 = k_ext_layer("m.self_attn.q_proj", per_block=2)
        e2, d2 = k_ext_layer("m.self_attn.k_proj", per_block=4)
        spec = parse_residue_metadata(
            make_metadata(
                {"m.self_attn.q_proj": e1, "m.self_attn.k_proj": e2},
                {**d1, **d2},
            )
        )
        with pytest.raises(ResidueMetadataError, match="conflicting"):
            spec.spec_for("m.self_attn.qkv_proj")

    def test_linear_attention_qkvz_resolves_through_split_parts(self):
        layers, dims = {}, {}
        for part in ("in_proj_qkv", "in_proj_z"):
            name = f"m.linear_attn.{part}"
            entry, d = k_ext_layer(name, per_block=1)
            layers[name] = entry
            dims.update(d)
        spec = parse_residue_metadata(make_metadata(layers, dims))
        fused = spec.spec_for("m.linear_attn.in_proj_qkvz")
        assert fused is not None
        assert fused.mode is ResidueMode.K_EXT
        assert fused.name == "m.linear_attn.in_proj_qkvz"
        assert fused.ratio == pytest.approx(0.125)

    def test_linear_attention_qkvz_rejects_conflicting_split_parts(self):
        qkv, qkv_dims = k_ext_layer("m.linear_attn.in_proj_qkv", per_block=1)
        z, z_dims = k_ext_layer("m.linear_attn.in_proj_z", per_block=2)
        spec = parse_residue_metadata(
            make_metadata(
                {
                    "m.linear_attn.in_proj_qkv": qkv,
                    "m.linear_attn.in_proj_z": z,
                },
                {**qkv_dims, **z_dims},
            )
        )
        with pytest.raises(ResidueMetadataError, match="conflicting"):
            spec.spec_for("m.linear_attn.in_proj_qkvz")


class TestRejection:
    def test_missing_runtime_mode(self):
        entry, dims = k_ext_layer("l")
        del entry["runtime_mode"]
        with pytest.raises(ResidueMetadataError, match="runtime_mode"):
            parse_residue_metadata(make_metadata({"l": entry}, dims))

    def test_unknown_runtime_mode(self):
        entry, dims = k_ext_layer("l")
        entry["runtime_mode"] = "wext_r2"
        with pytest.raises(ResidueMetadataError, match="unknown runtime_mode"):
            parse_residue_metadata(make_metadata({"l": entry}, dims))

    def test_unsupported_ratio(self):
        # 3 per 8-channel block = ratio 0.375, not a supported kernel
        entry, dims = k_ext_layer("l", per_block=3)
        with pytest.raises(ResidueMetadataError, match="not supported"):
            parse_residue_metadata(make_metadata({"l": entry}, dims))

    def test_k_ext_without_extended_dim(self):
        entry, _ = k_ext_layer("l")
        with pytest.raises(ResidueMetadataError, match="extended_linear_dims"):
            parse_residue_metadata(make_metadata({"l": entry}))

    def test_k_ext_num_salient_mismatch(self):
        entry, dims = k_ext_layer("l")
        entry["num_salient"] = entry["num_salient"] - 1
        with pytest.raises(ResidueMetadataError, match="num_salient"):
            parse_residue_metadata(make_metadata({"l": entry}, dims))

    def test_k_ext_unsorted_indices(self):
        entry, dims = k_ext_layer("l")
        entry["salient_indices"][0], entry["salient_indices"][1] = (
            entry["salient_indices"][1],
            entry["salient_indices"][0],
        )
        with pytest.raises(ResidueMetadataError, match="strictly increasing"):
            parse_residue_metadata(make_metadata({"l": entry}, dims))

    def test_k_ext_out_of_range_indices(self):
        entry, dims = k_ext_layer("l")
        entry["salient_indices"][-1] = K + 5
        with pytest.raises(ResidueMetadataError, match="out of range"):
            parse_residue_metadata(make_metadata({"l": entry}, dims))

    def test_mext_r1_with_extended_dim_conflicts(self):
        with pytest.raises(ResidueMetadataError, match="mext_r1"):
            parse_residue_metadata(make_metadata({"l": mext_r1_layer()}, {"l": 2 * K}))

    def test_mext_r1_with_partial_indices(self):
        entry = mext_r1_layer()
        entry["salient_indices"] = entry["salient_indices"][:-1]
        entry["num_salient"] = len(entry["salient_indices"])
        with pytest.raises(ResidueMetadataError):
            parse_residue_metadata(make_metadata({"l": entry}))

    def test_standard_with_indices_contradicts(self):
        entry, dims = k_ext_layer("l")
        entry["runtime_mode"] = "standard"
        with pytest.raises(ResidueMetadataError, match="contradicts"):
            parse_residue_metadata(make_metadata({"l": entry}, dims))

    def test_moe_active_impl_rejected(self):
        with pytest.raises(ResidueMetadataError, match="residue MoE"):
            parse_residue_metadata(
                make_metadata(
                    {"l": mext_r1_layer()},
                    moe={
                        "granularity": "layer",
                        "layers": {
                            "model.layers.0.mlp": {"impl": "mext_r1", "ratio": 1.0}
                        },
                    },
                )
            )

    def test_unsupported_block_size(self):
        data = make_metadata({"l": mext_r1_layer()})
        data["global"]["block_size"] = 16
        with pytest.raises(ResidueMetadataError, match="block_size"):
            parse_residue_metadata(data)

    def test_orphaned_extended_dim(self):
        with pytest.raises(ResidueMetadataError, match="without a valid residue"):
            parse_residue_metadata(
                make_metadata({"l": mext_r1_layer()}, {"ghost_layer": 2 * K})
            )

    def test_root_not_object(self):
        with pytest.raises(ResidueMetadataError, match="root"):
            parse_residue_metadata(["not", "a", "dict"])

    def test_missing_layers_section(self):
        with pytest.raises(ResidueMetadataError, match="layers"):
            parse_residue_metadata({"global": {}})


class TestFileLoading:
    def test_absent_file_returns_none(self, tmp_path):
        assert load_residue_model_spec(tmp_path) is None

    def test_valid_file_loads(self, tmp_path):
        entry, dims = k_ext_layer("l")
        (tmp_path / METADATA_FILENAME).write_text(
            json.dumps(make_metadata({"l": entry}, dims))
        )
        spec = load_residue_model_spec(tmp_path)
        assert spec is not None
        assert spec.spec_for("l").mode is ResidueMode.K_EXT

    def test_malformed_json_raises_not_falls_back(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text("{not json")
        with pytest.raises(ResidueMetadataError, match="cannot read"):
            load_residue_model_spec(tmp_path)

    def test_broken_contract_raises_not_falls_back(self, tmp_path):
        entry, dims = k_ext_layer("l")
        del entry["runtime_mode"]
        (tmp_path / METADATA_FILENAME).write_text(
            json.dumps(make_metadata({"l": entry}, dims))
        )
        with pytest.raises(ResidueMetadataError):
            load_residue_model_spec(tmp_path)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
