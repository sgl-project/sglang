from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams, TOP_K_ALL
from sglang.srt.speculative.dflash_request_validation import (
    validate_dflash_request_options,
)
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _req(**sampling_overrides):
    sampling_params = SimpleNamespace(
        sampling_seed=None,
        top_k=TOP_K_ALL,
        top_p=1.0,
        min_p=0.0,
        json_schema=None,
        regex=None,
        ebnf=None,
        structural_tag=None,
    )
    for key, value in sampling_overrides.items():
        setattr(sampling_params, key, value)
    return SimpleNamespace(
        return_logprob=False,
        return_hidden_states=False,
        sampling_params=sampling_params,
    )


class _FakeTimeStats:
    def set_tokenize_finish_time(self):
        pass


def test_seeded_greedy_sampling_is_allowed():
    assert (
        validate_dflash_request(
            _req(sampling_seed=1234, top_k=1), enable_overlap=False
        )
        is None
    )


def test_seeded_non_greedy_sampling_is_allowed():
    assert (
        validate_dflash_request_options(
            return_logprob=False,
            return_hidden_states=False,
            sampling_params=_req(sampling_seed=1234, top_k=TOP_K_ALL).sampling_params,
            enable_overlap=False,
            sampling_backend="pytorch",
        )
        is None
    )


def test_seeded_non_greedy_sampling_rejects_unseeded_backend():
    assert (
        validate_dflash_request_options(
            return_logprob=False,
            return_hidden_states=False,
            sampling_params=_req(sampling_seed=1234, top_k=TOP_K_ALL).sampling_params,
            enable_overlap=False,
            sampling_backend="flashinfer",
        )
        == "DFLASH speculative decoding with seeded non-greedy sampling "
        "requires --sampling-backend pytorch unless deterministic inference is enabled."
    )


def test_seeded_non_greedy_sampling_is_allowed_in_deterministic_mode():
    assert (
        validate_dflash_request_options(
            return_logprob=False,
            return_hidden_states=False,
            sampling_params=_req(sampling_seed=1234, top_k=TOP_K_ALL).sampling_params,
            enable_overlap=False,
            enable_deterministic_inference=True,
            sampling_backend="flashinfer",
        )
        is None
    )


def test_seeded_filtered_sampling_rejects_flashinfer_backend_even_in_deterministic_mode():
    assert (
        validate_dflash_request_options(
            return_logprob=False,
            return_hidden_states=False,
            sampling_params=_req(
                sampling_seed=1234, top_k=TOP_K_ALL, top_p=0.95
            ).sampling_params,
            enable_overlap=False,
            enable_deterministic_inference=True,
            sampling_backend="flashinfer",
        )
        == "DFLASH speculative decoding with seeded filtered sampling "
        "requires --sampling-backend pytorch."
    )


def test_unseeded_non_greedy_sampling_is_allowed():
    assert validate_dflash_request(_req(top_k=TOP_K_ALL), enable_overlap=False) is None


def test_tokenizer_manager_allows_seeded_non_greedy(monkeypatch):
    monkeypatch.setattr("sglang.srt.managers.tokenizer_manager.use_mlx", lambda: False)
    manager = SimpleNamespace(
        preferred_sampling_params=None,
        sampling_params_class=SamplingParams,
        tokenizer=None,
        model_config=SimpleNamespace(vocab_size=32000),
        server_args=SimpleNamespace(
            speculative_algorithm="DSPARK",
            disable_overlap_schedule=False,
            enable_deterministic_inference=False,
            sampling_backend="pytorch",
            disaggregation_transfer_backend=None,
        ),
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
        rid_to_state={None: SimpleNamespace(time_stats=_FakeTimeStats())},
    )
    obj = GenerateReqInput(
        input_ids=[1, 2, 3],
        sampling_params={
            "temperature": 0.7,
            "top_k": -1,
            "sampling_seed": 1234,
        },
    )

    tokenized = TokenizerManager._create_tokenized_object(
        manager,
        obj,
        input_text=None,
        input_ids=[1, 2, 3],
    )
    assert tokenized.sampling_params.sampling_seed == 1234
