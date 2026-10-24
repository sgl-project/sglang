"""Native CPU types/allocation checks, not Simulator performance acceptance."""

import importlib.util

import pytest
from test_cpu_bootstrap import run_python

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("sglang") is None,
    reason="requires a native SGLang/PyTorch CPU environment",
)


def test_native_result_next_input_and_cpu_reserve_use_native_sglang_types():
    run_python(
        """
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from sglang_simulator.simulation.sglang.hook_bootstrap import install_simulator_hooks
install_simulator_hooks()
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang_simulator.simulation.sglang.dspark import *
from sglang_simulator.simulation.sglang.dspark_runtime import NativeDSparkResults, prepare_cpu_decode
r = DSparkRequest('a', 512, 6, 384, 512, 1, 20, '{"max_new_tokens":20}')
g = DSparkGroup(0, 'TARGET_VERIFY', (r,), '{"path":"CPU fixture"}')
s = DSparkStep(0, DSparkTopology(2,1), (g,), '{"source":"CPU fixture"}')
t = DSparkStepTiming(0, tuple(DSparkPhaseCost(p, .01, 'CPU fixture') for p in s.required_phases))
p = DSparkProgress(0, (('a', 2),), 'CPU fixture')
prediction = DSparkStepResult(s,t,p)
result = NativeDSparkResults(1).build(batch=NS(seq_lens=torch.tensor([512])), prediction=prediction)
assert isinstance(result, GenerationBatchResult)
assert isinstance(result.next_draft_input, DFlashDraftInputV2)
assert result.next_token_ids.tolist() == [1] * 6
assert result.accept_lens.tolist() == [2]
assert result.block_accept_lens is None and result.cap_lens is None
assert result.new_seq_lens.tolist() == [514]
assert result.next_draft_input.new_seq_lens.tolist() == [514]
assert result.next_draft_input.bonus_tokens.tolist() == [1]
pool = ReqToTokenPool(1, 32, 'cpu', False)
pool.write((1, slice(0,4)), torch.tensor([1,2,3,4], dtype=torch.int32))
req = NS(kv=NS(kv_committed_len=4, kv_allocated_len=4), decode_batch_idx=0, sampling_params=NS(top_k=100))
batch = NS(device='cpu', reqs=[req], req_pool_indices=torch.tensor([1]),
           req_to_token_pool=pool, token_to_kv_pool_allocator=NS(page_size=1),
           tree_cache=object(), maybe_evict_swa=lambda:None)
with patch('sglang.srt.mem_cache.allocation.alloc_token_slots', return_value=torch.arange(5,17,dtype=torch.int64)) as allocate:
    result.next_draft_input.prepare_for_decode(batch)
    allocate.assert_called_once_with(batch.tree_cache, 12)
assert pool.req_to_token[1,:16].tolist() == list(range(1,17))
assert req.kv.kv_allocated_len == 16 and req.kv.kv_committed_len == 4
assert batch.seq_lens_cpu.tolist() == [4]
assert result.next_draft_input.nxt_kv_lens_cpu.tolist() == [16]
assert req.decode_batch_idx == 1
""",
        bootstrap=True,
        cpu=True,
    )
