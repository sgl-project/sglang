"""Native Nemotron-H regression for Ascend Mamba2 dispatch and state layout."""

import math

import pytest

import sglang
from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=300, suite="full-test-1-npu-a3", nightly=True)


def _native_config():
    return NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        layers_block_type=["mamba", "attention", "mlp"],
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        mamba_num_heads=8,
        mamba_n_groups=1,
        n_groups=1,
        mamba_head_dim=64,
    )


@pytest.mark.parametrize("disable_cuda_graph", [True, False], ids=["eager", "graph"])
def test_native_mamba2_prefill_and_decode(tmp_path, disable_cuda_graph):
    # No VoiceChat code or downloaded weights: exercise the native model entry.
    config = _native_config()
    config.save_pretrained(tmp_path)
    engine = sglang.Engine(
        model_path=str(tmp_path),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        attention_backend="ascend",
        context_length=512,
        max_running_requests=1,
        max_total_tokens=512,
        mem_fraction_static=0.25,
        disable_cuda_graph=disable_cuda_graph,
        cuda_graph_max_bs_decode=1,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=-1,
        random_seed=419,
    )
    try:
        state = engine.get_server_info()["internal_states"][0]
        assert state["decode_cuda_graph_bs"] == ([] if disable_cuda_graph else [1])
        for input_ids in ([80, 444, 55, 187], [560, 409, 405, 871, 82, 406]):
            response = engine.generate(
                input_ids=input_ids,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
                return_logprob=True,
            )
            assert len(response["output_ids"]) == 4
            assert response["meta_info"]["prompt_tokens"] == len(input_ids)
            logprobs = response["meta_info"]["output_token_logprobs"]
            assert len(logprobs) == 4
            assert all(math.isfinite(row[0]) for row in logprobs)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
