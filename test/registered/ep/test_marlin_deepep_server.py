"""Prefill/decode smoke test using a small local AWQ MoE and TP=EP=2.

The model uses dummy weights: this checks execution, not language quality.
"""

import argparse
import json
import tempfile
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


def run(mode):
    import sglang as sgl

    with tempfile.TemporaryDirectory(prefix="marlin-deepep-model-") as directory:
        config = dict(
            architectures=["Qwen3MoeForCausalLM"],
            model_type="qwen3_moe",
            hidden_size=4096,
            intermediate_size=512,
            moe_intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            norm_topk_prob=True,
            hidden_act="silu",
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=128,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            bos_token_id=1,
            eos_token_id=2,
            quantization_config=dict(
                quant_method="awq",
                bits=4,
                group_size=128,
                zero_point=True,
                version="gemm",
            ),
        )
        Path(directory, "config.json").write_text(json.dumps(config))
        engine = sgl.Engine(
            model_path=directory,
            load_format="dummy",
            skip_tokenizer_init=True,
            dtype="bfloat16",
            quantization="awq_marlin",
            tp_size=2,
            ep_size=2,
            moe_runner_backend="marlin",
            moe_a2a_backend="deepep",
            deepep_mode=mode,
            disable_shared_experts_fusion=True,
            attention_backend="triton",
            mem_fraction_static=0.15,
            max_total_tokens=1024,
            context_length=256,
            cuda_graph_bs_decode=[1, 2, 4],
            cuda_graph_backend_prefill="disabled",
            disable_overlap_schedule=True,
            log_level="warning",
        )
        try:
            for prompts in ([[1, 3, 5, 7]], [[1, 4, 6], [1, 8, 9, 10, 11]]):
                output = engine.generate(
                    input_ids=prompts,
                    sampling_params=dict(
                        temperature=0, max_new_tokens=4, ignore_eos=True
                    ),
                )
                assert len(output) == len(prompts)
                for result in output:
                    assert len(result["output_ids"]) == 4, result
            print(f"Marlin + DeepEP {mode}: prefill/decode smoke passed", flush=True)
        finally:
            engine.shutdown()


@pytest.mark.parametrize("mode", ["normal", "low_latency", "auto"])
def test_server(mode):
    run(mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["normal", "low_latency", "auto"], default=None
    )
    parser.add_argument("-f", action="store_true")  # CI's fail-fast flag.
    mode = parser.parse_args().mode
    if mode is None:
        raise SystemExit(pytest.main([__file__, "-v", "-s"]))
    run(mode)
