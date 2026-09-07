from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


class DotsNoteOmniTokenizerProxy:
    @classmethod
    def from_pretrained(cls, model_path: str, *args, **kwargs):
        kwargs.pop("use_fast", None)
        return AutoTokenizer.from_pretrained(model_path, *args, **kwargs)


@register_customized_processor(DotsNoteOmniTokenizerProxy)
class Dots3Config(PretrainedConfig):
    model_type = "dots3_note"
    keys_to_ignore_at_inference = ["past_key_values"]
    is_hybrid_swa = True
    requires_draft_attention_wrapper = True

    def __init__(
        self,
        # General model parameters
        vocab_size=152064,
        hidden_size=2560,
        hidden_act="silu",
        intermediate_size=7168,
        num_hidden_layers=30,
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pretraining_tp=1,
        # Token IDs
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151645,
        tie_word_embeddings=False,
        # Attention parameters
        attention_bias=False,
        attention_dropout=0.0,
        apply_mla_qkv_lora_rescale=True,
        # MLA (Multi-head Latent Attention) parameters
        attention_gate_type="headwise",
        kv_lora_rank=512,
        q_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        num_attention_heads=64,
        num_key_value_heads=64,
        v_head_dim=128,
        # Dots3 uses one shared MTP layer for NEXTN decoding.
        num_nextn_predict_layers=1,
        # Sliding Window Attention (SWA) parameters
        layer_types=None,
        sliding_window_size=512,
        swa_attention_gate_type="headwise",
        swa_q_lora_rank=512,
        swa_kv_lora_rank=512,
        swa_qk_nope_head_dim=128,
        swa_qk_rope_head_dim=64,
        swa_rope_theta=None,
        swa_num_attention_heads=32,
        swa_num_key_value_heads=32,
        swa_v_head_dim=128,
        # MoE (Mixture of Experts) parameters
        moe_intermediate_size=1024,
        n_shared_experts=1,
        n_routed_experts=128,
        num_experts_per_tok=6,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        n_group=1,
        topk_method="noaux_tc",
        topk_group=1,
        # RoPE parameters
        rope_theta=50000.0,
        rope_scaling=None,
        # Optional DSA indexer parameters.
        index_n_heads=None,
        index_head_dim=None,
        index_topk=None,
        language_only=False,
        # Multimodal special tokens
        im_start_token="<|img|>",
        im_token="<|imgpad|>",
        im_end_token="<|endofimg|>",
        audio_start_token="<|audio_comp_start|>",
        audio_token="<|audio_comp_pad|>",
        audio_end_token="<|audio_comp_end|>",
        video_token="<|video_pad|>",
        **kwargs,
    ):
        # General model parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp

        # Attention parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.apply_mla_qkv_lora_rescale = apply_mla_qkv_lora_rescale

        # MLA (Multi-head Latent Attention) parameters
        self.attention_gate_type = attention_gate_type
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.v_head_dim = v_head_dim

        # MTP / NextN
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Sliding Window Attention (SWA) parameters
        self.layer_types = layer_types
        self.sliding_window_size = sliding_window_size
        self.swa_attention_gate_type = swa_attention_gate_type
        self.swa_q_lora_rank = swa_q_lora_rank
        self.swa_kv_lora_rank = swa_kv_lora_rank
        self.swa_qk_nope_head_dim = swa_qk_nope_head_dim
        self.swa_qk_rope_head_dim = swa_qk_rope_head_dim
        self.swa_rope_theta = rope_theta if swa_rope_theta is None else swa_rope_theta
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_v_head_dim = swa_v_head_dim
        # Runtime cache geometry for the SWA attention path.
        self.swa_head_dim = swa_qk_nope_head_dim + swa_qk_rope_head_dim

        # MoE (Mixture of Experts) parameters
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_method = topk_method
        self.topk_group = topk_group

        # RoPE parameters
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        # NSA (Native Sparse Attention) parameters
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.language_only = language_only

        self.im_start_token = im_start_token
        self.im_token = im_token
        self.im_end_token = im_end_token
        self.audio_start_token = audio_start_token
        self.audio_token = audio_token
        self.audio_end_token = audio_end_token
        # The chat template renders a video content part as this single token,
        # which the processor replaces with the flattened frames and audio.
        self.video_token = video_token

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def configure_draft_model(self) -> str:
        """Configure the recursively shared MTP layer with SWA geometry."""
        self.num_nextn_predict_layers = 1
        self.layer_types = ["sliding_attention"]
        self.attention_gate_type = self.swa_attention_gate_type
        self.kv_lora_rank = self.swa_kv_lora_rank
        self.q_lora_rank = self.swa_q_lora_rank
        self.qk_nope_head_dim = self.swa_qk_nope_head_dim
        self.qk_rope_head_dim = self.swa_qk_rope_head_dim
        self.num_attention_heads = self.swa_num_attention_heads
        self.num_key_value_heads = self.swa_num_key_value_heads
        self.v_head_dim = self.swa_v_head_dim
        return "Dots3NoteForCausalLMNextN"

    def wrap_attention_backend(self, runner, full_attn_backend):
        from sglang.srt.layers.attention.dots_hybrid_backend import (
            wrap_dots_attention_backend,
        )

        return wrap_dots_attention_backend(runner, full_attn_backend)

    def wrap_draft_decode_attention_backend(self, backend):
        from sglang.srt.layers.attention.dots_hybrid_backend import (
            wrap_dots_draft_decode_backend,
        )

        return wrap_dots_draft_decode_backend(backend)

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise ValueError(
                f"`rope_scaling` must be a dictionary, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
            "linear",
            "dynamic",
            "yarn",
        ]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic', 'yarn'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, (int, float))
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a number > 1, got {rope_scaling_factor}"
            )
