from types import SimpleNamespace

import pytest
import torch

from sglang.srt.beam_search.batch_tail import BeamTail, BeamTailEntry
from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.logits_capture import capture_pre_sample_logits
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOKEN_START = 1000
CODEBOOK = 16
BEAM = 8
PROMPT_LEN = 5
NUM_GROUPS = 2
VOCAB = TOKEN_START + 3 * CODEBOOK + 7


class _Req(SimpleNamespace):
    """A Req stand-in that BeamGroup can hold weakly, as it holds a real Req."""


class _ReqToTokenPool:
    def __init__(self):
        self.device = torch.device("cuda")
        self.req_to_token = torch.arange(
            64 * 32, dtype=torch.int32, device="cuda"
        ).view(64, 32)
        self._next_row = NUM_GROUPS

    def alloc_rows(self, n):
        rows = list(range(self._next_row, self._next_row + n))
        self._next_row += n
        return rows

    def available_size(self):
        return 64 - self._next_row


class _Storage:
    """The KV storage calls member spawn makes: taking request rows."""

    def __init__(self, pool):
        self.take_rows = pool.alloc_rows


class _FutureMap:
    def __init__(self):
        self.tokens = torch.zeros(64, dtype=torch.int64, device="cuda")

    def stash(self, rows, payload):
        self.tokens[rows] = payload.bonus_tokens


def _setup():
    def full(*shape):
        return torch.ones(shape, dtype=torch.bool, device="cuda")

    trie = BeamTrieConstraint(
        config=TrieOutputHeadConfig(
            tensor_path="",
            token_start=TOKEN_START,
            codebook_size=CODEBOOK,
            num_codebooks=3,
        ),
        root_mask=full(CODEBOOK),
        level1_masks=full(CODEBOOK, CODEBOOK),
        level2_masks=full(CODEBOOK * CODEBOOK, CODEBOOK),
        level2_sparse=None,
        level_cardinalities=(CODEBOOK, CODEBOOK * CODEBOOK, CODEBOOK**3),
    )
    pool = _ReqToTokenPool()
    coordinator = BeamCoordinator(
        model_config=SimpleNamespace(beam_trie_config=None),
        spec_algorithm=None,
        dllm_enabled=False,
        max_req_len=0,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(storage=_Storage(pool)),
        tree_cache=None,
        future_map=_FutureMap(),
        trie_constraint=trie,
    )
    reqs = []
    for row in range(NUM_GROUPS):
        group = BeamGroup(
            beam_width=BEAM, max_new_tokens=3, device="cuda", device_history=True
        )
        group.trie_prefix_codes = torch.empty((1, 0), dtype=torch.int64, device="cuda")
        group.prompt_len = PROMPT_LEN
        req = _Req(
            beam_group=group,
            kv=ReqKvInfo(
                req_pool_idx=row,
                kv_committed_len=PROMPT_LEN,
                kv_allocated_len=PROMPT_LEN,
            ),
            output_ids=[],
        )
        group.leader = req
        reqs.append(req)
    return coordinator, reqs


def _forward(mode, rows, width, with_normalizer):
    generator = torch.Generator(device="cuda").manual_seed(rows + width)
    logits = torch.randn((rows, width), generator=generator, device="cuda")
    output = LogitsProcessorOutput(next_token_logits=logits.to(torch.bfloat16))
    if with_normalizer:
        output.beam_normalizer = torch.randn(
            (rows, 2), generator=generator, device="cuda"
        ).abs()
    forward_batch = SimpleNamespace(
        forward_mode=mode, positions=torch.zeros(rows, device="cuda")
    )
    return output, forward_batch


def _run_request_steps(coordinator, reqs):
    # Prefill: the leaders' first selection and member spawn.
    batch = SimpleNamespace(beam_tail=None, forward_mode=ForwardMode.EXTEND, reqs=reqs)
    output, forward_batch = _forward(ForwardMode.EXTEND, NUM_GROUPS, VOCAB, False)
    capture_pre_sample_logits(batch, forward_batch, output)
    for pos, req in enumerate(reqs):
        coordinator.select_leader_prefill(req, pos, output, tick=0)

    # Two decode steps, the second final, as the batch tail lays out the rows.
    for tick in (1, 2):
        for req in reqs:
            req.kv.kv_committed_len = PROMPT_LEN + tick
        members = BEAM - 1
        batch = SimpleNamespace(
            beam_tail=BeamTail(
                num_base_rows=NUM_GROUPS,
                entries=[
                    BeamTailEntry(req.beam_group, i, i * members, (i + 1) * members)
                    for i, req in enumerate(reqs)
                ],
            ),
            forward_mode=ForwardMode.DECODE,
            reqs=reqs,
            forward_iter=tick,
        )
        output, forward_batch = _forward(
            ForwardMode.DECODE, NUM_GROUPS * BEAM, CODEBOOK, True
        )
        capture_pre_sample_logits(batch, forward_batch, output)
        coordinator.select_and_relay_decode(batch, output)


def test_beam_launch_half_never_syncs_the_host():
    """Run-ahead: from the forward's logits to the relayed tokens, the beam
    launch half (capture, selection, member spawn, KV remap, relay) issues no
    host sync, so the CPU queues it while the forward still runs."""
    # Warm-up: kernel compilation and first allocations.
    _run_request_steps(*_setup())
    torch.cuda.synchronize()
    coordinator, reqs = _setup()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _run_request_steps(coordinator, reqs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for req in reqs:
        group = req.beam_group
        assert group.num_generated == 3
        assert len(group._pending_steps) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
