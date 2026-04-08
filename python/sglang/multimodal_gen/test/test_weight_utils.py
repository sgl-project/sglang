import json
import pytest
import torch

from sglang.multimodal_gen.runtime.loader.weight_utils import (
    VAE_CHECKSUM_EXCLUDE_NAMES,
    compute_weights_checksum,
    compute_weights_checksum_content_only,
    filter_weights_for_checksum,
    get_disk_to_model_weights,
    get_module_arch_from_disk,
)


class TestComputeWeightsChecksum:
    def test_checksum_deterministic(self):
        params = [("x", torch.randn(2, 3)), ("y", torch.randn(1, 4))]
        assert compute_weights_checksum(params) == compute_weights_checksum(params)

    def test_checksum_depends_on_names(self):
        t = torch.randn(2, 3)
        assert compute_weights_checksum([("a", t)]) != compute_weights_checksum([("b", t)])

    def test_content_only_ignores_names(self):
        t = torch.randn(2, 3)
        assert compute_weights_checksum_content_only([("a", t)]) == compute_weights_checksum_content_only([("b", t)])

    def test_content_only_same_contents_same_checksum(self):
        t = torch.randn(2, 3)
        u = torch.randn(1, 2)
        c1 = compute_weights_checksum_content_only([("x", t), ("y", u)])
        c2 = compute_weights_checksum_content_only([("other", t.clone()), ("z", u.clone())])
        assert c1 == c2


class TestGetDiskToModelWeights:
    def test_identity_unknown_arch(self):
        disk = [("layer.0.weight", torch.randn(4, 4))]
        out = list(get_disk_to_model_weights("UnknownArch", disk))
        assert out == disk

    def test_qwen3_strip_model_prefix(self):
        disk = [("model.layers.0.norm.weight", torch.randn(8))]
        out = list(get_disk_to_model_weights("Qwen3ForCausalLM", disk))
        assert len(out) == 1 and out[0][0] == "layers.0.norm.weight" and out[0][1].shape == (8,)

    def test_qwen3_skip_rotary(self):
        disk = [
            ("model.layers.0.self_attn.rotary_emb.inv_freq", torch.randn(8)),
            ("model.embed_tokens.weight", torch.randn(100, 256)),
        ]
        out = list(get_disk_to_model_weights("Qwen3ForCausalLM", disk))
        assert len(out) == 1 and out[0][0] == "embed_tokens.weight"

    def test_qwen3_merge_qkv(self):
        q, k, v = torch.randn(32, 256), torch.randn(8, 256), torch.randn(8, 256)
        disk = [
            ("model.layers.0.self_attn.q_proj.weight", q),
            ("model.layers.0.self_attn.k_proj.weight", k),
            ("model.layers.0.self_attn.v_proj.weight", v),
        ]
        out = list(get_disk_to_model_weights("FSDPQwen3ForCausalLM", disk))
        assert len(out) == 1
        assert out[0][0] == "layers.0.self_attn.qkv_proj.weight"
        merged = out[0][1]
        assert merged.shape == (48, 256)
        assert torch.allclose(merged[:32], q) and torch.allclose(merged[32:40], k) and torch.allclose(merged[40:48], v)

    def test_qwen3_merge_gate_up(self):
        gate, up = torch.randn(64, 256), torch.randn(192, 256)
        disk = [
            ("model.layers.0.mlp.gate_proj.weight", gate),
            ("model.layers.0.mlp.up_proj.weight", up),
        ]
        out = list(get_disk_to_model_weights("Qwen3ForCausalLM", disk))
        assert len(out) == 1 and out[0][0] == "layers.0.mlp.gate_up_proj.weight" and out[0][1].shape == (256, 256)


class TestVAEChecksumExclude:
    """VAE BatchNorm buffers (bn.*) are not in disk safetensors; exclude from checksum."""

    def test_exclude_names_defined(self):
        assert "bn.num_batches_tracked" in VAE_CHECKSUM_EXCLUDE_NAMES
        assert "bn.running_mean" in VAE_CHECKSUM_EXCLUDE_NAMES
        assert "bn.running_var" in VAE_CHECKSUM_EXCLUDE_NAMES

    def test_filter_weights_removes_bn_params(self):
        weights = [
            ("bn.num_batches_tracked", torch.zeros(1, dtype=torch.long)),
            ("bn.running_mean", torch.randn(4)),
            ("bn.running_var", torch.randn(4)),
            ("encoder.conv_in.weight", torch.randn(8, 4, 3, 3)),
        ]
        filtered = list(filter_weights_for_checksum(weights, VAE_CHECKSUM_EXCLUDE_NAMES))
        assert len(filtered) == 1 and filtered[0][0] == "encoder.conv_in.weight"

    def test_checksum_same_with_or_without_bn_buffers(self):
        """Checksum of (real params only) should match (real params + bn buffers) after filter."""
        real = [("a", torch.randn(2, 3)), ("b", torch.randn(1, 4))]
        with_bn = real + [
            ("bn.num_batches_tracked", torch.zeros(1, dtype=torch.long)),
            ("bn.running_mean", torch.randn(4)),
            ("bn.running_var", torch.randn(4)),
        ]
        c_real = compute_weights_checksum(real)
        c_filtered = compute_weights_checksum(
            filter_weights_for_checksum(with_bn, VAE_CHECKSUM_EXCLUDE_NAMES)
        )
        assert c_real == c_filtered


class TestGetModuleArchFromDisk:
    def test_nonexistent_path(self, tmp_path):
        assert get_module_arch_from_disk(str(tmp_path), "no_such") is None

    def test_missing_config(self, tmp_path):
        (tmp_path / "text_encoder").mkdir()
        assert get_module_arch_from_disk(str(tmp_path), "text_encoder") is None

    def test_read_arch(self, tmp_path):
        (tmp_path / "text_encoder").mkdir()
        (tmp_path / "text_encoder" / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
        assert get_module_arch_from_disk(str(tmp_path), "text_encoder") == "Qwen3ForCausalLM"
