# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.kv_shard_hooks import (
    get_kv_shard_pool,
    prepare_kv_shard_forward,
)
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveKVPoolMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingPool(PageInterleaveKVPoolMixin):
    def __init__(self):
        self.begin_args = None
        self.begin_calls = 0
        self.end_calls = 0

    def begin_shard_extend(self, *args):
        self.begin_args = args
        self.begin_calls += 1

    def end_shard_extend(self):
        self.end_calls += 1


def _batch(mode: ForwardMode):
    return SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=torch.tensor([3], dtype=torch.int64),
        extend_prefix_lens_cpu=[64],
        seq_lens_cpu=torch.tensor([128], dtype=torch.int64),
    )


def test_detects_only_page_interleaved_pools():
    pool = _RecordingPool()

    assert get_kv_shard_pool(pool) is pool
    assert get_kv_shard_pool(object()) is None


@pytest.mark.parametrize(
    "mode, active",
    [
        (ForwardMode.EXTEND, True),
        (ForwardMode.MIXED, True),
        (ForwardMode.SPLIT_PREFILL, True),
        (ForwardMode.DECODE, False),
        (ForwardMode.IDLE, False),
        (ForwardMode.TARGET_VERIFY, False),
        (ForwardMode.DRAFT_EXTEND_V2, False),
    ],
)
def test_prepare_updates_the_pool_lifecycle(mode, active):
    pool = _RecordingPool()
    req_to_token = torch.arange(8)
    batch = _batch(mode)

    assert prepare_kv_shard_forward(pool, req_to_token, batch) is active
    assert pool.begin_calls == int(active)
    assert pool.end_calls == int(not active)
    if active:
        assert all(
            actual is expected
            for actual, expected in zip(
                pool.begin_args,
                (
                    req_to_token,
                    batch.req_pool_indices,
                    batch.extend_prefix_lens_cpu,
                    batch.seq_lens_cpu,
                ),
            )
        )
    else:
        assert pool.begin_args is None


@pytest.mark.parametrize(
    "missing_field",
    ["req_pool_indices", "extend_prefix_lens_cpu", "seq_lens_cpu"],
)
def test_extend_requires_host_metadata(missing_field):
    batch = _batch(ForwardMode.EXTEND)
    setattr(batch, missing_field, None)

    with pytest.raises(RuntimeError, match="requires request indices and CPU"):
        prepare_kv_shard_forward(_RecordingPool(), torch.empty(0), batch)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
