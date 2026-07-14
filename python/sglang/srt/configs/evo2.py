# SPDX-License-Identifier: Apache-2.0
"""Evo2 model configuration for SGLang.

Evo2 (StripedHyena 2) is a hybrid DNA foundation model that interleaves
standard attention layers with Hyena convolution operators (HCL, HCM, HCS).
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Evo2Config(PretrainedConfig):
    """Configuration for the Evo2 StripedHyena 2 model.

    Supports the following variants:
    - evo2-1b-8k
    - evo2-7b-8k / evo2-7b-1m
    """

    model_type = "evo2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 4096,
        num_filters: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 11008,
        hidden_act: str = "gelu",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 32768,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        # Hyena-specific parameters
        state_size: int = 16,
        short_filter_length: int = 3,
        hcm_filter_length: int = 128,
        hcs_filter_length: int = 7,
        hcl_filter_groups: int = 4096,
        hcm_filter_groups: int = 256,
        hcs_filter_groups: int = 256,
        # Layer indices
        attn_layer_idxs: list = None,
        hcl_layer_idxs: list = None,
        hcm_layer_idxs: list = None,
        hcs_layer_idxs: list = None,
        # RoPE
        rotary_emb_base: float = 10000.0,
        rotary_emb_scaling_factor: float = None,
        use_interpolated_rotary_pos_emb: bool = False,
        # Evo2-specific
        evo2_style_activations: bool = True,
        mlp_activation: str = "gelu",
        interleave: bool = True,
        column_split: bool = True,
        column_split_hyena: bool = False,
        # Bias flags
        mha_out_proj_bias: bool = True,
        hyena_out_proj_bias: bool = True,
        hyena_flip_x1x2: bool = False,
        qkv_proj_bias: bool = False,
        short_filter_bias: bool = False,
        # Tokenizer
        tokenizer_type: str = "CharLevelTokenizer",
        make_vocab_size_divisible_by: int = 8,
        inner_size_multiple_of: int = 16,
        # FP8
        use_fp8_input_projections: bool = False,
        # Flash attention / FFT
        use_flash_attn: bool = True,
        use_flashfft: bool = False,
        use_flash_depthwise: bool = False,
        use_flash_rmsnorm: bool = False,
        # Misc
        final_norm: bool = True,
        inference_mode: bool = True,
        prefill_style: str = "fft",
        tie_embeddings: bool = True,
        proj_groups: int = 1,
        model_parallel_size: int = 1,
        pipe_parallel_size: int = 1,
        print_activations: bool = False,
        log_intermediate_values: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache

        # Hyena parameters
        self.state_size = state_size
        self.short_filter_length = short_filter_length
        self.hcm_filter_length = hcm_filter_length
        self.hcs_filter_length = hcs_filter_length
        self.hcl_filter_groups = hcl_filter_groups
        self.hcm_filter_groups = hcm_filter_groups
        self.hcs_filter_groups = hcs_filter_groups

        # Default layer indices for evo2-7b-8k (32 layers)
        if attn_layer_idxs is None:
            attn_layer_idxs = [3, 10, 17, 24, 31]
        if hcl_layer_idxs is None:
            hcl_layer_idxs = [2, 6, 9, 13, 16, 20, 23, 27, 30]
        if hcm_layer_idxs is None:
            hcm_layer_idxs = [1, 5, 8, 12, 15, 19, 22, 26, 29]
        if hcs_layer_idxs is None:
            hcs_layer_idxs = [0, 4, 7, 11, 14, 18, 21, 25, 28]

        self.attn_layer_idxs = attn_layer_idxs
        self.hcl_layer_idxs = hcl_layer_idxs
        self.hcm_layer_idxs = hcm_layer_idxs
        self.hcs_layer_idxs = hcs_layer_idxs

        # RoPE
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scaling_factor = rotary_emb_scaling_factor
        self.use_interpolated_rotary_pos_emb = use_interpolated_rotary_pos_emb

        # Evo2 flags
        self.evo2_style_activations = evo2_style_activations
        self.mlp_activation = mlp_activation
        self.interleave = interleave
        self.column_split = column_split
        self.column_split_hyena = column_split_hyena

        # Bias flags
        self.mha_out_proj_bias = mha_out_proj_bias
        self.hyena_out_proj_bias = hyena_out_proj_bias
        self.hyena_flip_x1x2 = hyena_flip_x1x2
        self.qkv_proj_bias = qkv_proj_bias
        self.short_filter_bias = short_filter_bias

        # Other
        self.tokenizer_type = tokenizer_type
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.inner_size_multiple_of = inner_size_multiple_of
        self.use_fp8_input_projections = use_fp8_input_projections
        self.use_flash_attn = use_flash_attn
        self.use_flashfft = use_flashfft
        self.use_flash_depthwise = use_flash_depthwise
        self.use_flash_rmsnorm = use_flash_rmsnorm
        self.final_norm = final_norm
        self.inference_mode = inference_mode
        self.prefill_style = prefill_style
        self.tie_embeddings = tie_embeddings
        self.proj_groups = proj_groups
        self.model_parallel_size = model_parallel_size
        self.pipe_parallel_size = pipe_parallel_size
        self.print_activations = print_activations
        self.log_intermediate_values = log_intermediate_values

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    @num_hidden_layers.setter
    def num_hidden_layers(self, value: int):
        self.num_layers = value


# ──────────────────────────────────────────────────────────────────────────────
# YAML-derived defaults for each Evo 2 variant
# Keys match vortex model YAML config names.
# ──────────────────────────────────────────────────────────────────────────────

# Fields shared across ALL variants (overridden per-variant below)
_EVO2_BASE_DEFAULTS = {
    "vocab_size": 512,
    "short_filter_length": 3,
    "hcm_filter_length": 128,
    "hcs_filter_length": 7,
    "state_size": 16,
    "rotary_emb_base": 10000.0,
    "short_filter_bias": False,
    "rms_norm_eps": 1e-6,
    "make_vocab_size_divisible_by": 8,
    "inner_size_multiple_of": 16,
    "proj_groups": 1,
    "hyena_filter_groups": 1,
    "column_split_hyena": False,
    "column_split": True,
    "interleave": True,
    "evo2_style_activations": True,
    "model_parallel_size": 1,
    "pipe_parallel_size": 1,
    "tie_embeddings": True,
    "mha_out_proj_bias": True,
    "hyena_out_proj_bias": True,
    "hyena_flip_x1x2": False,
    "qkv_proj_bias": False,
    "final_norm": True,
    "use_flash_attn": True,
    "use_flash_rmsnorm": False,
    "use_flash_depthwise": False,
    "use_flashfft": False,
    "use_laughing_hyena": False,
    "inference_mode": True,
    "tokenizer_type": "CharLevelTokenizer",
    "prefill_style": "fft",
    "mlp_activation": "gelu",
    "max_batch_size": 1,
    "print_activations": False,
    "log_intermediate_values": False,
}

# Per-variant overrides
_EVO2_VARIANT_DEFAULTS = {
    "1b": {
        "hidden_size": 1920,
        "num_filters": 1920,
        "num_layers": 25,
        "num_attention_heads": 15,
        "num_key_value_heads": 15,
        "intermediate_size": 5120,
        "hcl_filter_groups": 1920,
        "hcm_filter_groups": 128,
        "hcs_filter_groups": 128,
        "attn_layer_idxs": [3, 10, 17, 24],
        "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23],
        "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22],
        "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21],
        "max_position_embeddings": 8192,
        "use_fp8_input_projections": True,
    },
    "7b": {
        "hidden_size": 4096,
        "num_filters": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "hcl_filter_groups": 4096,
        "hcm_filter_groups": 256,
        "hcs_filter_groups": 256,
        "attn_layer_idxs": [3, 10, 17, 24, 31],
        "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23, 27, 30],
        "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22, 26, 29],
        "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21, 25, 28],
    },
    "7b-8k": {
        "intermediate_size": 11008,
        "max_position_embeddings": 32768,
        "use_fp8_input_projections": False,
    },
    "7b-262k": {
        "intermediate_size": 11008,
        "max_position_embeddings": 262144,
        "use_fp8_input_projections": True,
        "use_interpolated_rotary_pos_emb": True,
        "rotary_emb_scaling_factor": 32,
    },
    "7b-1m": {
        "intermediate_size": 11264,
        "max_position_embeddings": 1048576,
        "use_fp8_input_projections": True,
        "use_interpolated_rotary_pos_emb": True,
        "rotary_emb_scaling_factor": 128,
    },
}


def _detect_evo2_variant(model: str) -> str:
    """Detect Evo2 variant from model name or path.

    Raises ValueError for unrecognised variants.
    """
    name = str(model).lower()
    if "1b" in name:
        return "1b"
    if "262k" in name:
        return "7b-262k"
    if "1m" in name:
        return "7b-1m"
    if "7b" in name:
        # evo2_7b (no suffix) is the 1M-context flagship;
        # evo2_7b_base is the 8k variant.
        if "base" in name:
            return "7b-8k"
        return "7b-1m"
    raise ValueError(
        f"Unknown Evo 2 variant for '{model}'. "
        f"Expected a name containing '1b', '7b', '262k', or '1m'."
    )


def patch_evo2_config_json(model: str) -> None:
    """Auto-generate a complete config.json for Evo 2 Hub checkpoints.

    Some Evo2 repos ship with config.yml instead of config.json.
    This handles both, detects the Evo2 variant, and writes a complete
    config.json with all fields from the Vortex YAML configs.

    original configs can be found here:
    https://github.com/ArcInstitute/evo2/tree/main/evo2/configs

    """
    import json
    import os

    import yaml

    cfg_path = os.path.join(model, "config.json")
    cfg = None

    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        yml_path = os.path.join(model, "config.yml")
        if os.path.isfile(yml_path):
            with open(yml_path) as f:
                cfg = yaml.safe_load(f)
        else:
            # Try HuggingFace Hub download
            try:
                from huggingface_hub import hf_hub_download

                for fname in ("config.json", "config.yml"):
                    try:
                        cfg_path = hf_hub_download(model, fname)
                        with open(cfg_path) as f:
                            cfg = (
                                json.load(f)
                                if fname.endswith(".json")
                                else yaml.safe_load(f)
                            )
                        break
                    except Exception:
                        continue
            except Exception:
                pass

    if cfg is None:
        # No config file exists at all — generate from variant defaults alone
        variant = _detect_evo2_variant(str(model))
        cfg = {"_name_or_path": str(model)}

    # Detect if this is an Evo2 model
    architectures = cfg.get("architecture") or cfg.get("architectures") or []
    model_name = str(model).lower()
    cfg_model_name = cfg.get("model_name", "") or cfg.get("name", "") or ""
    is_evo2 = (
        "StripedHyena2" in architectures
        or cfg_model_name.startswith("shc-evo2")
        or "evo2" in model_name
        or "evo-2" in model_name
    )
    if not is_evo2:
        return

    variant = _detect_evo2_variant(
        cfg.get("name", "")
        or cfg.get("_name_or_path", "")
        or cfg.get("model_name", "")
        or str(model)
    )

    # Build complete defaults
    defaults = dict(_EVO2_BASE_DEFAULTS)
    base_key = "7b" if "7b" in variant else "1b"
    defaults.update(_EVO2_VARIANT_DEFAULTS[base_key])
    defaults.update(_EVO2_VARIANT_DEFAULTS.get(variant, {}))

    # Merge: existing config values take priority
    patched = dict(defaults)
    patched.update(cfg)
    patched["model_type"] = "evo2"
    patched["architectures"] = ["Evo2ForCausalLM"]
    # Ensure correct token IDs from the auto-generated CharLevelTokenizer
    patched.setdefault("pad_token_id", 1)
    patched.setdefault("bos_token_id", 0)
    patched.setdefault("eos_token_id", 0)
    # Enforce bfloat16: fp16 overflows in the IIR FFT at deep HCL layers
    # where the pre-norm scale reaches ~5.5 (layer 23 in 7B).
    patched.setdefault("torch_dtype", "bfloat16")

    # Always write config.json
    out_path = os.path.join(model, "config.json")
    tmp_path = out_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(patched, f, indent=2)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
