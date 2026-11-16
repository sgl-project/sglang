"""CPU request state implementing the current prefix-cache protocol."""

from array import array
from dataclasses import fields
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ReqKvInfo


class RequestState(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "req_to_token" in kwargs and "write" not in kwargs:
            self.write = lambda index, value: self.req_to_token.__setitem__(
                index, value
            )
        if "cache_request_handle" not in kwargs:
            return
        self.extra_key = kwargs.get("extra_key")
        self.cache_salt = kwargs.get("cache_salt")
        self.full_untruncated_fill_ids = kwargs.get(
            "full_untruncated_fill_ids",
            kwargs.get(
                "fill_ids",
                kwargs.get("get_fill_ids", lambda: array("q"))()
                if "origin_input_ids" not in kwargs
                else array(
                    "q",
                    [
                        *kwargs.get("origin_input_ids", []),
                        *kwargs.get("output_ids", []),
                    ],
                ),
            ),
        )
        self.prefix_len = kwargs.get(
            "prefix_len", len(kwargs.get("expected_prefix", []))
        )
        self.finished = kwargs.get("finished", lambda: True)
        self.priority = kwargs.get("priority", 0)
        self.lock = kwargs.get("lock", None)
        old_kv = kwargs.get("kv", SimpleNamespace())
        names = {field.name for field in fields(ReqKvInfo)}
        self.kv = (
            old_kv
            if isinstance(old_kv, ReqKvInfo)
            else ReqKvInfo(
                **{name: value for name, value in vars(old_kv).items() if name in names}
            )
        )
        if hasattr(old_kv, "swa_evicted_seqlen"):
            self.kv.set_evicted_seqlen("swa", old_kv.swa_evicted_seqlen)


def discard_row(cache, req, length):
    from unittest.mock import Mock, patch

    from sglang.srt.mem_cache.common import release_kv_cache

    req.owned_kv_len = lambda: length
    req.skip_radix_cache_insert = False
    req.finished_len = None
    req.kv.kv_allocated_len = length
    req.kv.req_pool_idx = 0
    cache.token_to_kv_pool_allocator.target_window_pool.return_value = "swa"
    with (
        patch("sglang.srt.mem_cache.common.kv_storage_of", return_value=Mock()),
        patch(
            "sglang.srt.mem_cache.common.get_spec",
            return_value=SimpleNamespace(speculative_algorithm=None),
        ),
        patch(
            "sglang.srt.mem_cache.common.get_serving",
            return_value=SimpleNamespace(strip_thinking_cache=False),
        ),
    ):
        release_kv_cache(req, cache, checkpoint=False)
