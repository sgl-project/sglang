### used for TensorRT LLM

```
{
    "architecture": "LlamaForCausalLM",
    "dtype": "float16",
    "logits_dtype": "float32",
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "hidden_size": 16384,
    "num_hidden_layers": 126,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "head_size": 128,
    "qk_layernorm": false,
    "hidden_act": "silu",
    "intermediate_size": 53248,
    "norm_epsilon": 1e-05,
    "position_embedding_type": "rope_gpt_neox",
    "use_parallel_embedding": false,
    "embedding_sharding_dim": 0,
    "share_embedding_table": false,
    "mapping": {
        "world_size": 8,
        "tp_size": 8,
        "pp_size": 1,
        "gpus_per_node": 8
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": null,
        "group_size": 128,
        "smoothquant_val": null,
        "has_zero_point": false,
        "pre_quant_scale": false,
        "exclude_modules": [
            "lm_head"
        ]
    },
    "kv_dtype": "float16",
    "rotary_scaling": null,
    "residual_mlp": false,
    "moe_normalization_mode": null,
    "rotary_base": 500000.0,
    "moe_num_experts": 0,
    "moe_top_k": 0,
    "moe_tp_mode": 2,
    "attn_bias": false,
    "disable_weight_only_quant_plugin": false,
    "mlp_bias": false
}
```

### used for vLLM and SGLang

```
{
  "_name_or_path": "dummy_fp8",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "hidden_act": "silu",
  "hidden_size": 16384,
  "initializer_range": 0.02,
  "intermediate_size": 53248,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 128,
  "num_hidden_layers": 126,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "quantization_config": {
    "activation_scheme": "static",
    "ignored_layers": [
      "lm_head"
    ],
    "quant_method": "fp8"
  },
  "rope_scaling": {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "max_position_embeddings": 131072,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.41.1",
  "use_cache": true,
  "vocab_size": 128256
}
```
