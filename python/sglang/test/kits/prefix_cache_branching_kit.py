import requests


class PrefixCacheBranchingMixin:
    cache_chunk_size: int

    @classmethod
    def send_request_helper(cls, text: str):
        response = requests.post(
            cls.base_url + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "max_new_tokens": 1,
                },
            },
        )
        return response.json()

    @classmethod
    def test_prefix_cache_branching(cls):
        cls.flush_cache()
        branching_pos = 257
        text_prefix = "hi" * branching_pos
        suffix_list = [
            "this" * cls.cache_chunk_size * 4,
            "here" * cls.cache_chunk_size * 4,
            "that" * cls.cache_chunk_size * 4,
        ]
        cache_hit_list = [False, False, True]

        # First request only prefill the entire sequence
        # Second request won't have cache hit, but will cache the branching point
        # Third request will have cache hit on the branching point
        for i, (suffix, cache_hit) in enumerate(
            zip(suffix_list, cache_hit_list, strict=True)
        ):
            result = cls.send_request_helper(text_prefix + suffix)
            cached_tokens = result["meta_info"]["cached_tokens"]
            if cache_hit:
                expected_cached_tokens = (
                    branching_pos // cls.cache_chunk_size * cls.cache_chunk_size
                )
                assert (
                    cached_tokens == expected_cached_tokens
                ), f"{i=}, {cache_hit=}, {cached_tokens=} is not equal to {expected_cached_tokens=}, {branching_pos=}"
            else:
                assert (
                    cached_tokens == 0
                ), f"{i=}, {cache_hit=}, {cached_tokens=} is not 0"
