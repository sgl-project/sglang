# SPDX-License-Identifier: Apache-2.0
"""CPU correctness pins for OmniDreams (no checkpoint, no GPU required).

Two kinds of pins live here:

* **Phase-0 scaffold** — constructs the DiT on the meta device (no memory for
  ~2B params) and validates the checkpoint-exact structure against an
  independently-derived authoritative key fixture, plus the pre/post-fusion
  shapes, the 2-step flow-match sigmas, and the 3D-RoPE layout.
* **Regression guards** — each test pins a specific defect found while
  validating OmniDreams end-to-end on GPU:

  1. Meta-init load materializes the non-persistent sinusoidal ``emb`` buffer.
  2. ``TimestepEmbedding`` casts the float32 sinusoid to the MLP param dtype.
  3. The registry resolves a non-diffusers local checkpoint via a path
     detector, gated to dirs WITHOUT model_index.json.
  4. ``_compute_num_chunks`` maps ``num_frames`` -> chunk count and caps the AR
     rollout length (``_MAX_AR_CHUNKS``).
  5. ``apply_chat_template`` BatchEncoding output is normalized to input_ids.
  6. ``read_vae_state_dict`` reads diffusers safetensors; ``load_wan_vae``
     raises a helpful error for a non-diffusers state dict.
  7. ``OmniDreamsTextEncoderConfig._resolve_bf16_src`` prefers a local
     ``text_encoder`` dir, else the pinned HF id + revision.
"""

import json
import os
import types
from collections import OrderedDict
from itertools import chain

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.omnidreams import (
    OmniDreamsDiTArchConfig,
    OmniDreamsDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
    OmniDreamsPipelineConfig,
    warp_flow_match_sigmas,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
    OmniDreamsDiT,
    TimestepEmbedding,
    Timesteps,
    rope_dims,
)
from sglang.multimodal_gen.runtime.models.encoders.omnidreams_text import (
    COSMOS_REASON1_HIDDEN,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
    _MAX_AR_CHUNKS,
    _TEXT_MAX_LENGTH,
    OmniDreamsBeforeDenoisingStage,
)

_KEY_FIXTURE = os.path.join(
    os.path.dirname(__file__), "data", "omnidreams_dit_keys.txt"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tiny_arch() -> OmniDreamsDiTArchConfig:
    return OmniDreamsDiTArchConfig(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=4,
    )


def _load_fixture_keys() -> set[str]:
    with open(_KEY_FIXTURE) as f:
        return {line.strip() for line in f if line.strip()}


def _build_meta_model() -> OmniDreamsDiT:
    with torch.device("meta"):
        return OmniDreamsDiT(config=OmniDreamsDiTConfig(), hf_config={})


# --------------------------------------------------------------------------- #
# Phase-0 scaffold: structure / shape / schedule / RoPE layout
# --------------------------------------------------------------------------- #
def test_state_dict_matches_authoritative_key_fixture():
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    model = _build_meta_model()
    keys = set(model.state_dict().keys())
    ckpt_keys = _load_fixture_keys()
    assert len(ckpt_keys) == 570
    # The packed-QKV merge maps the checkpoint's separate q/k/v -> to_qkv and k/v ->
    # to_kv (param_names_mapping). Apply it to the checkpoint keys and verify they
    # cover exactly the model's state_dict (i.e. the flat checkpoint stays loadable).
    mapping_fn = get_param_names_mapping(OmniDreamsDiT.param_names_mapping)
    mapped = {mapping_fn(k)[0] for k in ckpt_keys}
    assert (
        mapped == keys
    ), f"missing={sorted(keys - mapped)} extra={sorted(mapped - keys)}"


def test_unique_bias_is_crossattn_proj():
    model = _build_meta_model()
    biases = [k for k in model.state_dict() if k.endswith(".bias")]
    assert biases == ["crossattn_proj.0.bias"]


def test_pre_fusion_shapes():
    model = _build_meta_model()
    sd = model.state_dict()
    # x_embedder keeps the padding-mask channel pre-fusion: (16 + 1 + 1) * 2 * 2 = 72.
    assert tuple(sd["x_embedder.proj.1.weight"].shape) == (2048, 72)
    # HDMap embed: 16 * 2 * 2 = 64 in-features.
    assert tuple(sd["additional_patch_embedding.proj.1.weight"].shape) == (2048, 64)
    # Final layer pre-shuffle: patch_dim = 2*2*1*16 = 64.
    assert tuple(sd["final_layer.linear.weight"].shape) == (64, 2048)
    assert tuple(sd["crossattn_proj.0.weight"].shape) == (1024, 100352)


def test_post_load_weights_fuses_in_place():
    model = _build_meta_model()
    pre_keys = set(model.state_dict().keys())
    model.post_load_weights()
    sd = model.state_dict()
    # Padding-mask channels dropped: 72 -> 68.
    assert tuple(sd["x_embedder.proj.1.weight"].shape) == (2048, 68)
    # Shuffle fuse is a reorder; shape is preserved.
    assert tuple(sd["final_layer.linear.weight"].shape) == (64, 2048)
    # Fusion must not add or remove parameters.
    assert set(sd.keys()) == pre_keys
    assert model._is_padding_mask_fused and model._is_shuffle_op_fused


def test_two_step_flow_match_sigmas():
    sigmas = warp_flow_match_sigmas()
    assert len(sigmas) == 3
    assert abs(sigmas[0] - 1.0) < 1e-9
    assert abs(sigmas[1] - 0.8036) < 1e-3
    assert sigmas[2] == 0.0
    # The pipeline config exposes the same schedule.
    assert OmniDreamsPipelineConfig().denoising_sigmas() == sigmas


def test_rope_layout_neox_44_42_42():
    assert rope_dims(128) == (44, 42, 42)
    assert sum(rope_dims(128)) == 128


# --------------------------------------------------------------------------- #
# Regression 1: meta-init buffer materialization
# --------------------------------------------------------------------------- #
def test_meta_init_materializes_nonpersistent_buffers():
    with torch.device("meta"):
        model = OmniDreamsDiT(
            config=OmniDreamsDiTConfig(arch_config=_tiny_arch()), hf_config={}
        )
    # Simulate the production load path: materialize params/buffers on a real
    # device (mirrors the FSDP loader), then run the post-load hook.
    model.to_empty(device="cpu")
    model.post_load_weights()

    on_meta = [
        n
        for n, p in chain(model.named_parameters(), model.named_buffers())
        if p.is_meta
    ]
    assert not on_meta, f"params/buffers left on meta: {on_meta}"

    ts = model.t_embedder[0]
    assert isinstance(ts, Timesteps)
    assert ts.emb.device.type == "cpu"
    assert torch.isfinite(ts.emb).all()
    assert ts.emb.shape == (ts.num_channels // 2,)


# --------------------------------------------------------------------------- #
# Regression 2: TimestepEmbedding dtype cast
# --------------------------------------------------------------------------- #
def test_timestep_embedding_casts_sinusoid_to_param_dtype():
    te = TimestepEmbedding(16, 16, use_adaln_lora=True).to(torch.bfloat16)
    sinusoid_fp32 = torch.randn(16, dtype=torch.float32)
    raw, lora = te(sinusoid_fp32)  # must not raise float != bf16
    assert raw.dtype == torch.bfloat16  # raw embedding cast for RMSNorm/AdaLN
    assert lora.dtype == torch.bfloat16
    assert torch.isfinite(raw).all() and torch.isfinite(lora).all()


# --------------------------------------------------------------------------- #
# Regression 3: registry resolution + gated short-circuit
# --------------------------------------------------------------------------- #
def test_registry_resolves_nondiffusers_local_omnidreams(tmp_path):
    import sglang.multimodal_gen.registry as reg

    local = tmp_path / "omni-dreams"
    local.mkdir()  # non-diffusers: no model_index.json

    reg._get_config_info.cache_clear()
    info = reg._get_config_info(str(local))
    assert info is not None
    assert info.pipeline_config_cls.__name__ == "OmniDreamsPipelineConfig"


def test_registry_path_detector_gated_to_non_diffusers(tmp_path):
    import sglang.multimodal_gen.registry as reg

    # A diffusers-style dir (has model_index.json) with a neutral name and an
    # unknown _class_name must NOT be short-circuited by the path detectors;
    # it falls through to model_index resolution and returns None (no match),
    # proving step 3a did not fire for a model_index.json dir.
    d = tmp_path / "plain-model"
    d.mkdir()
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "ZzzUnknownPipeline"})
    )

    reg._get_config_info.cache_clear()
    info = reg._get_config_info(str(d))
    assert info is None


# --------------------------------------------------------------------------- #
# Regression 4: num_frames -> chunk mapping + AR cap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_frames,expected",
    [(5, 1), (6, 2), (13, 2), (14, 3), (21, 3)],  # len_t=2: first=5, step=8
)
def test_compute_num_chunks_boundaries(num_frames, expected):
    batch = types.SimpleNamespace(num_frames=num_frames)
    assert (
        OmniDreamsBeforeDenoisingStage._compute_num_chunks(batch, len_t=2) == expected
    )


def test_compute_num_chunks_caps_ar_loop():
    batch = types.SimpleNamespace(num_frames=10_000_000)
    assert (
        OmniDreamsBeforeDenoisingStage._compute_num_chunks(batch, len_t=2)
        == _MAX_AR_CHUNKS
    )


# --------------------------------------------------------------------------- #
# Regression 5: tokenizer BatchEncoding normalization
# --------------------------------------------------------------------------- #
def test_encode_text_normalizes_batchencoding():
    n_layers = 3  # tiny stand-in for the 28 transformer layers
    hidden = COSMOS_REASON1_HIDDEN

    class _DictTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            # Newer transformers return a BatchEncoding (dict-like), not a tensor.
            return {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

    def _text_encoder(
        input_ids=None,
        attention_mask=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # 1 embedding layer + n_layers transformer layers, each [B, L, H].
        L = input_ids.shape[1]
        hs = [torch.randn(1, L, hidden) for _ in range(n_layers + 1)]
        return types.SimpleNamespace(hidden_states=hs)

    stage = OmniDreamsBeforeDenoisingStage.__new__(OmniDreamsBeforeDenoisingStage)
    stage.tokenizer = _DictTokenizer()
    stage.text_encoder = _text_encoder
    stage._text_embed_cache = OrderedDict()  # __new__ bypasses __init__

    out = stage._encode_text("a prompt", torch.device("cpu"))
    assert out.shape == (1, _TEXT_MAX_LENGTH, n_layers * hidden)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Regression 6: VAE state-dict reader + helpful error
# --------------------------------------------------------------------------- #
def test_read_vae_state_dict_safetensors_file_and_dir(tmp_path):
    from safetensors.torch import save_file

    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        read_vae_state_dict,
    )

    tensors = {"w": torch.zeros(2, 3)}
    f = tmp_path / "vae.safetensors"
    save_file(tensors, str(f))

    sd_file = read_vae_state_dict(str(f))
    assert set(sd_file.keys()) == {"w"}

    d = tmp_path / "vae"
    d.mkdir()
    save_file(tensors, str(d / "diffusion_pytorch_model.safetensors"))
    sd_dir = read_vae_state_dict(str(d))
    assert set(sd_dir.keys()) == {"w"}

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_vae_state_dict(str(empty))


def test_load_wan_vae_raises_helpful_error_on_key_mismatch(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import sglang.multimodal_gen.runtime.models.vaes.wanvae as wanvae_mod
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        load_wan_vae,
    )

    class _FakeVAE:
        def __init__(self, config):
            pass

        def load_state_dict(self, state, strict=True):
            raise RuntimeError("Missing key(s) in state_dict: ...")

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(wanvae_mod, "AutoencoderKLWan", _FakeVAE)

    f = tmp_path / "vae.safetensors"
    save_file({"original_wan_key": torch.zeros(1)}, str(f))

    with pytest.raises(RuntimeError, match="diffusers format"):
        load_wan_vae(object(), str(f), torch.device("cpu"), torch.float32)


# --------------------------------------------------------------------------- #
# Regression 7: text-encoder source resolution
# --------------------------------------------------------------------------- #
def test_resolve_text_encoder_src_prefers_local_then_hf(tmp_path):
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsTextEncoderConfig,
    )

    cfg = OmniDreamsTextEncoderConfig(model_path=str(tmp_path))

    # No local text_encoder dir -> falls back to the pinned HF id + revision.
    src, rev = cfg._resolve_bf16_src()
    assert src == cfg.model_id and rev == cfg.revision

    # Local text_encoder/config.json present -> use the local dir, no revision.
    te = tmp_path / "text_encoder"
    te.mkdir()
    (te / "config.json").write_text("{}")
    src2, rev2 = cfg._resolve_bf16_src()
    assert src2 == str(te) and rev2 is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
