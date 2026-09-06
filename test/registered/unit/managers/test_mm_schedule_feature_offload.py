"""Raw multimodal features must leave the device once an item's embedding exists.

In chunked prefill the per-item split objects stay alive for every remaining
chunk of the request.  Their raw features are moved to the device for the
encoder; once the embedding exists the raw input is only needed as the
cache-miss fallback, so it goes back to host memory instead of sitting next to
the language model's activations.  That must hold for cache misses (the encode
sites in mm_schedule), for cache hits and for items deduplicated onto another
item's encode (the per-modality pass in mm_utils.embed_mm_inputs), and nothing
may keep the device tensor alive across the language model call.

Runs on CPU always: the encode paths, the no-op branch of the offload, and a
call spy that fails if any release site is dropped.  When CUDA is visible the
same scenarios additionally hold only a weak reference to each source tensor
and prove it is gone by the time the (fake) language model runs.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers import mm_schedule, mm_utils
from sglang.srt.managers.mm_schedule import PerImageRequestInfo
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")
# The weakref assertions only mean something with a real device allocation.
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 4
VOCAB = 16
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _device(name: str) -> torch.device:
    # Production takes the device from input_ids, so it carries an index.
    return torch.empty(0, device=name).device


def _feature(item_hash: int, tokens: int, device) -> torch.Tensor:
    rows = torch.arange(tokens * HIDDEN, dtype=torch.float32).reshape(tokens, HIDDEN)
    return (rows + item_hash * 1000).to(device)


def _encode(feature: torch.Tensor) -> torch.Tensor:
    """The fake encoder's transform: a fresh tensor, never a view of its input."""
    return feature * 2 + 1


def _item(item_hash: int, start: int, tokens: int, device="cpu") -> MultimodalDataItem:
    """A per-image item covering placeholder positions [start, start + tokens)."""
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(start, start + tokens - 1)],
        feature=_feature(item_hash, tokens, device),
    )
    item.set_hash(item_hash)
    return item


def _expected_rows(item: MultimodalDataItem) -> torch.Tensor:
    start, end = item.offsets[0]
    return _encode(_feature(item.hash, end - start + 1, "cpu"))


def _ids(items):
    return [id(item) for item in items]


def _fake_encoder(device: torch.device):
    """Encoder stand-in: requires device features, returns fresh embeddings.

    Records which items it saw (never their tensors), so the test's weak
    references are the only handle on the source features.
    """
    calls = []

    def encode(items):
        for item in items:
            assert (
                item.feature.device.type == device.type
            ), "feature must be on the encoder device when encoding"
        calls.append(list(items))
        return torch.cat([_encode(item.feature) for item in items], dim=0)

    encode.calls = calls
    return encode


def _refusing_encoder(items):
    pytest.fail("the encoder must not run when every item is a cache hit")


def _assert_on_host(items):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for item in items:
        assert isinstance(item.feature, torch.Tensor)
        assert item.feature.device.type == "cpu"


def _assert_source_released(refs, device: torch.device):
    # On CPU the source tensor *is* the host copy, so only the CUDA
    # parametrization can show the device allocation was let go.
    if device.type == "cuda":
        assert all(ref() is None for ref in refs), "device feature still referenced"


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(mm_schedule, "embedding_cache", MultiModalStaticCache(1 << 20))


@pytest.fixture
def runtime_stubs(monkeypatch):
    """general_mm_embed_routine reads runtime config no server published here."""
    monkeypatch.setattr(mm_utils, "get_server_args", lambda: None)
    monkeypatch.setattr(
        mm_utils, "get_disagg", lambda: SimpleNamespace(language_only=False)
    )


def _batch_inputs(requests, device):
    """input_ids / lengths / MultimodalInputs for one chunk covering every
    request whole.  ``requests`` is one item list per request."""
    seq_lens = []
    ids = []
    for items in requests:
        seq_len = max(end for item in items for _, end in item.offsets) + 1
        req_ids = torch.zeros(seq_len, dtype=torch.long)
        for item in items:
            start, end = item.offsets[0]
            req_ids[start : end + 1] = item.pad_value
        seq_lens.append(seq_len)
        ids.append(req_ids)
    mm_inputs = [MultimodalInputs(mm_items=list(items)) for items in requests]
    return torch.cat(ids).to(device), seq_lens, mm_inputs


class _FakeLanguageModel:
    """Runs the caller's checks where the prefill's own activations would be
    allocated, then hands input_embeds back as the hidden states."""

    def __init__(self, device, check):
        self._embed = torch.nn.Embedding(VOCAB, HIDDEN).to(device)
        self._check = check
        self.forward_calls = 0

    def get_input_embeddings(self):
        return self._embed

    def __call__(self, input_ids, forward_batch, input_embeds, **kwargs):
        self.forward_calls += 1
        self._check()
        return input_embeds


def _run_embed_routine(requests, encoder, device, check):
    """Drive general_mm_embed_routine (encode, scatter, offload, LM) for one
    chunk and return the input_embeds the language model received."""
    input_ids, seq_lens, mm_inputs = _batch_inputs(requests, device)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False
        ),
        contains_mm_inputs=lambda: True,
        mm_inputs=mm_inputs,
        extend_prefix_lens_cpu=[0] * len(requests),
        extend_seq_lens_cpu=seq_lens,
        input_embeds=None,
    )
    language_model = _FakeLanguageModel(device, check)
    out = mm_utils.general_mm_embed_routine(
        input_ids=input_ids,
        forward_batch=forward_batch,
        language_model=language_model,
        data_embedding_funcs={Modality.IMAGE: encoder},
    )
    assert language_model.forward_calls == 1
    return out


def _run_embed_mm_inputs(requests, encoder, device):
    input_ids, seq_lens, mm_inputs = _batch_inputs(requests, device)
    out, _ = mm_utils.embed_mm_inputs(
        mm_inputs_list=mm_inputs,
        extend_prefix_lens=[0] * len(requests),
        extend_seq_lens=seq_lens,
        input_ids=input_ids,
        input_embedding=torch.nn.Embedding(VOCAB, HIDDEN).to(device),
        data_embedding_func_mapping={Modality.IMAGE: encoder},
    )
    return out


def _assert_rows(out: torch.Tensor, req_start: int, item: MultimodalDataItem):
    start, end = item.offsets[0]
    rows = out[req_start + start : req_start + end + 1].cpu()
    torch.testing.assert_close(rows, _expected_rows(item), rtol=0, atol=0)


# --- the helper itself -------------------------------------------------------


def test_offload_items_to_host_only_touches_device_tensors():
    cpu_item = _item(1, 0, 2)
    cpu_feature = cpu_item.feature
    list_item = MultimodalDataItem(modality=Modality.IMAGE, feature=[1, 2, 3])
    none_item = MultimodalDataItem(modality=Modality.IMAGE, feature=None)

    mm_schedule._offload_items_to_host([cpu_item, list_item, none_item])

    assert cpu_item.feature is cpu_feature
    assert list_item.feature == [1, 2, 3]
    assert none_item.feature is None


@pytest.mark.parametrize("device", DEVICES)
def test_offload_items_to_host_skips_none_placeholders(device):
    """The final pass hands the helper the raw ``mm_items`` list, which may
    carry ``None`` placeholders that the inline loop it replaces tolerated."""
    item = _item(1, 0, 2, device)
    mm_schedule._offload_items_to_host([None, item, None])
    _assert_on_host([item])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_items_to_host_moves_cuda_features():
    item = _item(1, 0, 2, "cuda")
    ref = weakref.ref(item.feature)
    expected = _feature(1, 2, "cpu")

    mm_schedule._offload_items_to_host([item])

    _assert_on_host([item])
    assert ref() is None
    torch.testing.assert_close(item.feature, expected, rtol=0, atol=0)


# --- encode sites: released as soon as the encoder returns -------------------


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_embedding_full_releases_after_encode(fresh_cache, device):
    device = _device(device)
    items = [_item(1, 0, 3, device), _item(2, 3, 2, device)]
    refs = [weakref.ref(item.feature) for item in items]
    encoder = _fake_encoder(device)

    chunk, _ = mm_schedule._get_chunked_embedding_full(
        encoder,
        items,
        [(0, 2), (3, 4)],
        extend_prefix_len=0,
        extend_seq_len=5,
        input_ids=torch.zeros(5, dtype=torch.long, device=device),
        device=device,
    )

    assert [_ids(call) for call in encoder.calls] == [_ids(items)]
    _assert_on_host(items)
    _assert_source_released(refs, device)
    assert chunk.device.type == device.type
    expected = torch.cat([_expected_rows(item) for item in items], dim=0)
    torch.testing.assert_close(chunk.cpu(), expected, rtol=0, atol=0)
    cached = mm_schedule.embedding_cache.get([item.hash for item in items])
    assert cached is not None and cached.embedding.device.type == device.type


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_embedding_by_item_releases_after_encode(fresh_cache, device):
    device = _device(device)
    items = [_item(1, 0, 3, device), _item(2, 3, 2, device)]
    refs = [weakref.ref(item.feature) for item in items]
    encoder = _fake_encoder(device)

    # First chunk covers only the first image; the second is not encoded.
    chunk = mm_schedule._get_chunked_embedding_by_item(
        encoder,
        items,
        [(0, 2), (3, 4)],
        extend_prefix_len=0,
        extend_seq_len=3,
        device=device,
    )

    assert [_ids(call) for call in encoder.calls] == [_ids(items[:1])]
    _assert_on_host(items[:1])
    _assert_source_released(refs[:1], device)
    assert refs[1]() is items[1].feature
    torch.testing.assert_close(chunk.cpu(), _expected_rows(items[0]), rtol=0, atol=0)
    cached = mm_schedule.embedding_cache.get_single(items[0].hash)
    assert cached is not None and cached.embedding.device.type == device.type


@pytest.mark.parametrize("device", DEVICES)
def test_batch_encode_per_image_misses_releases_after_encode(fresh_cache, device):
    device = _device(device)
    items = [_item(1, 0, 3, device), _item(2, 3, 2, device)]
    refs = [weakref.ref(item.feature) for item in items]
    encoder = _fake_encoder(device)
    req_info = PerImageRequestInfo(
        req_idx=0,
        items=items,
        items_offset=[(0, 2), (3, 4)],
        extend_prefix_len=0,
        extend_seq_len=5,
    )

    hash_to_embedding = mm_schedule._batch_encode_per_image_misses(
        encoder, [req_info], device
    )

    assert [_ids(call) for call in encoder.calls] == [_ids(items)]
    _assert_on_host(items)
    _assert_source_released(refs, device)
    for item in items:
        want = _expected_rows(item)
        got = hash_to_embedding[(item.hash, want.shape[0])]
        assert got.device.type == device.type
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
        cached = mm_schedule.embedding_cache.get_single(item.hash)
        assert cached is not None and cached.embedding.device.type == device.type


# --- whole embedding step: nothing survives into the language model ----------


@pytest.mark.parametrize("device", DEVICES)
def test_cold_miss_released_before_language_model(fresh_cache, runtime_stubs, device):
    device = _device(device)
    items = [_item(1, 0, 3, device), _item(2, 4, 2, device)]
    refs = [weakref.ref(item.feature) for item in items]
    encoder = _fake_encoder(device)

    def check():
        _assert_on_host(items)
        _assert_source_released(refs, device)

    out = _run_embed_routine([items], encoder, device, check)

    assert [_ids(call) for call in encoder.calls] == [_ids(items)]
    for item in items:
        _assert_rows(out, 0, item)


@pytest.mark.parametrize("device", DEVICES)
def test_warm_cache_hit_released_before_language_model(
    fresh_cache, runtime_stubs, device
):
    device = _device(device)
    # Populate the cache the way production does: a first request encodes it.
    _run_embed_routine(
        [[_item(7, 0, 3, device)]], _fake_encoder(device), device, lambda: None
    )
    assert mm_schedule.embedding_cache.get_single(7) is not None

    # A later request for the same content arrives with its own device-resident
    # feature (transport reconstruction can place it on the GPU directly).
    item = _item(7, 0, 3, device)
    ref = weakref.ref(item.feature)

    def check():
        _assert_on_host([item])
        _assert_source_released([ref], device)

    out = _run_embed_routine([[item]], _refusing_encoder, device, check)

    _assert_rows(out, 0, item)


@pytest.mark.parametrize("device", DEVICES)
def test_same_hash_requests_released_before_language_model(
    fresh_cache, runtime_stubs, device
):
    device = _device(device)
    # Two requests in one batch carrying separate tensors for the same content;
    # only one of them is handed to the encoder.
    first, second = _item(5, 0, 3, device), _item(5, 0, 3, device)
    refs = [weakref.ref(first.feature), weakref.ref(second.feature)]
    encoder = _fake_encoder(device)

    def check():
        _assert_on_host([first, second])
        _assert_source_released(refs, device)

    out = _run_embed_routine([[first], [second]], encoder, device, check)

    assert [_ids(call) for call in encoder.calls] == [_ids([first])]
    _assert_rows(out, 0, first)
    _assert_rows(out, 3, second)


# --- call spy: the release sites must stay wired, CPU or not -----------------


def test_every_release_site_invokes_offload(fresh_cache, runtime_stubs, monkeypatch):
    device = _device("cpu")
    real_offload = mm_schedule._offload_items_to_host
    calls = []

    def spy(items):
        calls.append(list(items))
        real_offload(items)

    monkeypatch.setattr(mm_schedule, "_offload_items_to_host", spy)
    monkeypatch.setattr(mm_utils, "_offload_items_to_host", spy)

    def offloaded(item):
        return any(any(seen is item for seen in call) for call in calls)

    # Encode sites: exactly the items handed to the encoder come back through
    # the offload, as soon as that encode returns.
    items = [_item(1, 0, 3), _item(2, 3, 2)]
    calls.clear()
    mm_schedule._get_chunked_embedding_full(
        _fake_encoder(device),
        items,
        [(0, 2), (3, 4)],
        extend_prefix_len=0,
        extend_seq_len=5,
        input_ids=torch.zeros(5, dtype=torch.long),
        device=device,
    )
    assert [_ids(call) for call in calls] == [_ids(items)]

    items = [_item(3, 0, 3), _item(4, 3, 2)]
    calls.clear()
    mm_schedule._get_chunked_embedding_by_item(
        _fake_encoder(device),
        items,
        [(0, 2), (3, 4)],
        extend_prefix_len=0,
        extend_seq_len=3,
        device=device,
    )
    assert [_ids(call) for call in calls] == [_ids(items[:1])]

    items = [_item(5, 0, 3), _item(6, 3, 2)]
    calls.clear()
    mm_schedule._batch_encode_per_image_misses(
        _fake_encoder(device),
        [
            PerImageRequestInfo(
                req_idx=0,
                items=items,
                items_offset=[(0, 2), (3, 4)],
                extend_prefix_len=0,
                extend_seq_len=5,
            )
        ],
        device,
    )
    assert [_ids(call) for call in calls] == [_ids(items)]

    # Cache hit: no encode happens, so the per-modality pass in
    # embed_mm_inputs is the only place that can release the feature.
    hit = _item(5, 0, 3)
    calls.clear()
    _run_embed_mm_inputs([[hit]], _refusing_encoder, device)
    assert offloaded(hit)

    # Same-hash requests: the deduplicated item never reaches an encode site
    # and must still be released by the per-modality pass.
    first, second = _item(9, 0, 3), _item(9, 0, 3)
    encoder = _fake_encoder(device)
    calls.clear()
    _run_embed_mm_inputs([[first], [second]], encoder, device)
    assert [_ids(call) for call in encoder.calls] == [_ids([first])]
    assert offloaded(first) and offloaded(second)

    # Final pass of general_mm_embed_routine: every request's item list goes
    # through the helper once more, right before the language model runs.
    items_a, items_b = [_item(11, 0, 3)], [_item(12, 0, 2)]
    calls.clear()
    _run_embed_routine([items_a, items_b], _fake_encoder(device), device, lambda: None)
    final_calls = [_ids(call) for call in calls[-2:]]
    assert final_calls == [_ids(items_a), _ids(items_b)]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
