"""SGLANG_OPT_DSV41_TARGET_METADATA_EARLY: verify inputs and their metadata glue
graph staged on a side stream beside the draft graph (waiting only for the head), and
the runner gating that keeps every other configuration and variant on the main path."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    _GraphBucket,
)
from sglang.srt.layers.attention.graph_variants import (
    DSV41_CANDIDATE_FILTERED,
    DSV41_CANDIDATE_UNFILTERED,
    Dsv41CandidateGraphVariants,
)
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.early_verify_metadata import (
    STAGED_SLOTS,
    EarlyVerifyMetadata,
    VerifyInputSources,
)
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

KEY = (2, "TARGET_VERIFY", "TARGET_VERIFY", DSV41_CANDIDATE_UNFILTERED)
HEAD = object()  # the head's event the side stream waits for


def _registry(device) -> CudaGraphBufferRegistry:
    registry = CudaGraphBufferRegistry(
        device=torch.device(device), max_bs=4, max_num_tokens=12
    )
    for name in sorted(STAGED_SLOTS):
        tokens = name in ("positions", "out_cache_loc")
        registry.register_slot(
            GraphSlot(
                name,
                (lambda bs, mt: (mt,)) if tokens else (lambda bs, mt: (bs,)),
                torch.int64,
                axis="tokens" if tokens else "bs",
            )
        )
    return registry


def _sources(step: int, device="cpu") -> dict:
    return {
        "positions": torch.arange(6, device=device) + step,
        "out_cache_loc": torch.arange(6, device=device) + 100 + step,
        "req_pool_indices": torch.tensor([3, 1], device=device),
        "seq_lens": torch.tensor([10, 11], device=device) + step,
    }


def _variants(span=16):
    return Dsv41CandidateGraphVariants(
        graph_limits=((DSV41_CANDIDATE_UNFILTERED, span),),
        capture_labels=(DSV41_CANDIDATE_UNFILTERED, DSV41_CANDIDATE_FILTERED),
        verify_extra_tokens=3,
    )


def _staging_runner(calls, *, captured=lambda key: key[0] != 4):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner._early_verify_metadata = SimpleNamespace(
        stage=lambda **kw: calls.append(("stage", kw["key"], kw["after"])),
        take=lambda **kw: calls.append(("take", kw["key"])) or True,
        discard=lambda: calls.append(("discard", None)),
    )
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner.capture_bs = [1, 2, 4]
    runner.captured_req_width = 3
    runner.buffer_registry = None
    runner.attention_graph_variants = _variants()
    runner._metadata_glue = SimpleNamespace(is_captured=captured)
    return runner


def test_runner_stages_only_captured_unpadded_early_variant_keys():
    calls = []
    runner = _staging_runner(calls)

    def stage(bs, max_seq_len=10):
        return runner.stage_verify_metadata_early(
            seq_lens=torch.zeros(bs),
            req_pool_indices=torch.zeros(bs),
            out_cache_loc=torch.zeros(bs * 3),
            positions=torch.zeros(bs * 3),
            max_seq_len=max_seq_len,
            after=HEAD,
        )

    assert not stage(3)  # padded to 4 at replay
    assert not stage(4)  # glue graph not captured yet
    assert not stage(2, max_seq_len=14)  # 14 + 3 > 16: the filtered variant
    assert not stage(2, max_seq_len=None)  # unknown length: the filtered fallback
    assert stage(2, max_seq_len=13)
    key = runner._glue_key(
        2, str(ForwardMode.TARGET_VERIFY), DSV41_CANDIDATE_UNFILTERED
    )
    assert key[-1] == DSV41_CANDIDATE_UNFILTERED
    assert calls == [("discard", None)] * 4 + [("stage", key, HEAD)]


def test_runner_takes_only_an_unpadded_captured_replay():
    calls = []
    runner = _staging_runner(calls)
    fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    mode = str(ForwardMode.TARGET_VERIFY)

    def take(*, raw_bs=2, bs=2, is_ragged=False):
        key = runner._glue_key(bs, mode, DSV41_CANDIDATE_UNFILTERED)
        return runner._take_early_verify_metadata(
            fb, raw_bs=raw_bs, bs=bs, is_ragged=is_ragged, key=key
        )

    assert not take(raw_bs=1)
    assert not take(is_ragged=True)
    assert not take(raw_bs=4, bs=4)  # its glue graph is gone
    assert take()
    key = runner._glue_key(2, mode, DSV41_CANDIDATE_UNFILTERED)
    assert calls == [("discard", None)] * 3 + [("take", key)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_side_stream_work_is_visible_after_take():
    """Real streams: what the side stream wrote is what the main stream reads after
    ``take``, step after step, while the main stream is busy past the head event; a
    batch whose tensors are not the staged ones (even with equal values) or another
    key is refused, so the replay refills."""
    device = torch.device("cuda")
    registry = _registry(device)
    out = torch.zeros(2, dtype=torch.int64, device=device)
    seq_lens_buf = registry.get_slot("seq_lens").buffer

    class Glue:
        def replay(self, key):
            torch.mul(seq_lens_buf[:2], 2, out=out)

    early = EarlyVerifyMetadata(device=device, log_every=1000)
    busy = torch.randn(2048, 2048, device=device)
    head = torch.cuda.Event()
    for step in range(5):
        sources = _sources(step, device=device)
        head.record()
        for _ in range(4):
            busy = busy @ busy.T / 2048  # the "draft", after the head
        early.stage(
            glue=Glue(),
            key=KEY,
            registry=registry,
            sources=VerifyInputSources(**sources),
            after=head,
        )
        other = dict(sources, positions=sources["positions"].clone())
        if step == 1:
            assert not early.take(key=KEY, forward_batch=SimpleNamespace(**other))
            continue
        if step == 2:
            key = (4, *KEY[1:])
            assert not early.take(key=key, forward_batch=SimpleNamespace(**sources))
            continue
        assert early.take(key=KEY, forward_batch=SimpleNamespace(**sources))
        assert (out + 0).tolist() == [2 * (10 + step), 2 * (11 + step)]
        positions = registry.get_slot("positions").buffer[:6]
        assert positions.tolist() == list(range(step, step + 6))


# The DSv4 backend side: the captured early metadata is written in place.


class _Core:
    """Mirrors DSV4AttnMetadata.copy_: tensors copied in place, except
    swa_out_cache_loc, which copy_ assigns."""

    def __init__(self, page_table, swa_out_cache_loc):
        self.page_table = page_table
        self.swa_out_cache_loc = swa_out_cache_loc

    def copy_(self, other):
        self.page_table.copy_(other.page_table)
        self.swa_out_cache_loc = other.swa_out_cache_loc


class _Metadata:
    def __init__(self, page_table):
        # As the real full build: the SWA write locations are left to the in-graph op.
        self.core_attn_metadata = _Core(page_table, None)

    def copy_(self, other):
        self.core_attn_metadata.copy_(other.core_attn_metadata)


def _building_backend():
    def build(*, raw_metadata, online_c128_state_slot_offset, graph_backed):
        # In-graph semantics: no free-memory chunking of the indexer schedule.
        assert graph_backed
        return _Metadata(raw_metadata.clone())

    return SimpleNamespace(
        builds_verify_metadata_early=True,
        cuda_graph_metadata_of_bucket_and_bs={_GraphBucket.TARGET_VERIFY: {}},
        _early_verify_metadata_ids=set(),
        low_ratios=(1, 2),
        make_forward_metadata_from_raw_verify=build,
        online_c128_mtp=SimpleNamespace(state_slot_offset=lambda: 0),
        token_to_kv_pool=SimpleNamespace(
            request_window=None,
            swa_req_ring_size=None,
            translate_loc_from_full_to_swa=lambda loc: loc + 1000,
        ),
    )


def test_every_capture_and_replay_writes_the_first_captured_early_object(monkeypatch):
    """The install replaces the in-graph op that translates the step's write
    locations; every later install writes them into the tensor the graph reads. The
    filtered variant's Raw entry of the same key is never touched."""
    monkeypatch.setattr(
        capture_mode, "_capture_attention_variant", DSV41_CANDIDATE_UNFILTERED
    )
    backend = _building_backend()
    raw = object()
    backend.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.TARGET_VERIFY][2] = raw
    for step in range(3):
        assert DeepseekV4AttnBackend._uses_early_verify_metadata(
            backend, ragged_layout=None
        )
        DeepseekV4AttnBackend._install_early_verify_metadata(
            backend,
            graph_key=2,
            raw_metadata=torch.tensor([1, 2]) + step,
            out_cache_loc=torch.tensor([5, 6]) + step,
        )
        if step == 0:
            captured = backend.forward_metadata
            swa = captured.core_attn_metadata.swa_out_cache_loc
            assert swa.dtype == torch.int32 and swa.tolist() == [1005, 1006]
    bucket = backend.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.TARGET_VERIFY]
    assert bucket[(2, "early")] is captured and backend.forward_metadata is captured
    assert bucket[2] is raw
    assert backend._early_verify_metadata_ids == {id(captured)}
    assert captured.core_attn_metadata.page_table.tolist() == [3, 4]
    # The verify graphs read the captured write locations by address.
    assert captured.core_attn_metadata.swa_out_cache_loc is swa
    assert swa.tolist() == [1007, 1008]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
