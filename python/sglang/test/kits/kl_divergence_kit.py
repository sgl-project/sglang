from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper,
    test_input_output_logprobs_match_prefill_cache_hit_helper,
)


class KLDivergenceMixin:
    kl_div_thres: float
    kl_div_thres_decode: float | None = None
    kl_div_thres_prefill: float | None = None
    kl_div_max_samples: int = 32
    kl_div_prefill_max_new_tokens: int = 512
    kl_div_decode_max_new_tokens: int = 512

    @classmethod
    def _build_acc_thresholds(cls, threshold):
        """Build an ACC_THRESHOLDS dict compatible with kl_test_utils."""
        return {cls.model: {"kl_div": threshold}}

    @classmethod
    def test_input_output_logprobs_match_prefill_cache_hit(cls):
        test_input_output_logprobs_match_prefill_cache_hit_helper(
            base_url=cls.base_url,
            ACC_THRESHOLDS=cls._build_acc_thresholds(
                cls.kl_div_thres_prefill or cls.kl_div_thres
            ),
            model_name=cls.model,
            max_samples=cls.kl_div_max_samples,
            max_new_tokens=cls.kl_div_prefill_max_new_tokens,
        )

    @classmethod
    def test_input_output_logprobs_match_decode_cache_hit(cls):
        test_input_output_logprobs_match_decode_cache_hit_helper(
            base_url=cls.base_url,
            ACC_THRESHOLDS=cls._build_acc_thresholds(
                cls.kl_div_thres_decode or cls.kl_div_thres
            ),
            model_name=cls.model,
            max_samples=cls.kl_div_max_samples,
            max_new_tokens=cls.kl_div_decode_max_new_tokens,
        )
