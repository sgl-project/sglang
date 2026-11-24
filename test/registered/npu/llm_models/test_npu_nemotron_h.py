"""Native Nemotron-H regression for Ascend Mamba2 dispatch and state layout."""

import math

import pytest
import torch
from safetensors.torch import load_file

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


def test_streaming_mamba_rewind_matches_full_prefill(tmp_path):
    """Page alignment must not replay a tail on top of its post-tail state."""
    _native_config().save_pretrained(tmp_path)
    engine = sglang.Engine(
        model_path=str(tmp_path),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        attention_backend="ascend",
        page_size=128,
        context_length=512,
        max_running_requests=2,
        max_total_tokens=1024,
        mem_fraction_static=0.25,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=-1,
        enable_streaming_session=True,
        random_seed=419,
    )
    try:
        # Performance dummy weights suppress the recurrent contribution. Use
        # nontrivial deterministic weights so stale state changes the logits.
        exported = tmp_path / "exported"
        exported.mkdir()
        engine.save_sharded_model(path=str(exported), pattern=None, max_size=None)
        weights = {}
        for shard in exported.glob("*.safetensors"):
            weights.update(load_file(str(shard)))
        assert weights
        for name, value in weights.items():
            if name.endswith(".A"):
                value.fill_(-0.2)
            elif name.endswith(".dt_bias"):
                value.fill_(-2.0)
            elif name.endswith(".D") or ("norm" in name and name.endswith("weight")):
                value.fill_(1.0)
            elif name.endswith("bias"):
                value.zero_()
            else:
                value.mul_(100.0)
        assert engine.begin_weight_update()[0]
        try:
            result = engine.update_weights_from_tensor(
                list(weights.items()), load_format="direct"
            )
            assert result[0], result
        finally:
            assert engine.end_weight_update()[0]

        # Both turns commit position 129, then require a multi-position append:
        # one follows a multi-token generation, the other appends fresh input.
        for start, generated, suffix in ((128, 2, []), (129, 1, [18, 21])):
            prompt = [(i * 37 + 11) % 1024 for i in range(start)]
            sid = engine.open_session(capacity_of_str_len=512, streaming=True)
            sampling = dict(temperature=0.0, max_new_tokens=generated, ignore_eos=True)
            first = engine.generate(
                input_ids=prompt, sampling_params=sampling, session_params={"id": sid}
            )
            sampling = dict(temperature=0.0, max_new_tokens=1, ignore_eos=True)
            actual = engine.generate(
                input_ids=suffix,
                sampling_params=sampling,
                session_params={"id": sid},
                return_logprob=True,
                top_logprobs_num=1024,
            )
            engine.close_session(sid)
            reference = engine.generate(
                input_ids=prompt + first["output_ids"] + suffix,
                sampling_params=sampling,
                return_logprob=True,
                top_logprobs_num=1024,
            )
            assert actual["output_ids"] == reference["output_ids"]
            actual_scores, reference_scores = [
                torch.tensor(
                    [
                        row[0]
                        for row in sorted(
                            response["meta_info"]["output_top_logprobs"][0],
                            key=lambda row: row[1],
                        )
                    ]
                )
                for response in (actual, reference)
            ]
            torch.testing.assert_close(
                actual_scores, reference_scores, atol=0.02, rtol=0.002
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
