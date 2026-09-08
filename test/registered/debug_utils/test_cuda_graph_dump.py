"""CPU tests for the CUDA-graph-compatible dumping machinery.

Everything here runs without a GPU. The one thing a CPU cannot exercise is a
real `torch.cuda.graph()` capture, so the T1 rung of the `tap()` ladder is
reached by faking `_stream_is_capturing`; the rung *ordering*, the buffer
bookkeeping, the padding trim and the seam wrappers are all genuinely under
test. The GPU-only acceptance items (address stability across a real replay,
set-equality against a pure-eager run) live in the design doc, not here.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from sglang.srt.debug_utils.cuda_graph import (
    BufferKey,
    BufferRegistry,
    CudaGraphDumpConfig,
    DumpBudgetExceeded,
)
from sglang.srt.debug_utils.cuda_graph import seams as seams_module
from sglang.srt.debug_utils.cuda_graph import state as state_module
from sglang.srt.debug_utils.cuda_graph.registry import DumpBufferGrowthRejected
from sglang.srt.debug_utils.cuda_graph.seams import (
    _backend_label,
    _first_arg,
    _pd_role,
    _wrap,
    _wrap_forward_fn,
    install_backend_seam,
    install_runner_seam,
)
from sglang.srt.debug_utils.cuda_graph.state import (
    _CudaGraphDumpState,
    _row_pairs,
    _trim,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="weekly", runner_config="cpu")


class _ShapeToken(NamedTuple):
    """A hashable stand-in for the in-tree `ShapeKey`.

    The registry keys its per-buffer layout table by shape token, so the token
    has to be hashable — the real `ShapeKey` is a frozen dataclass and hashes by
    value. `types.SimpleNamespace` defines `__eq__` and therefore has no
    `__hash__`, which makes it unusable here even though it is fine for the
    runner / forward-batch stand-ins elsewhere in this file.
    """

    size: int


def _registry(budget_bytes: int = 1 << 30, prefixes=None, strict: bool = False):
    config = CudaGraphDumpConfig(
        enable=True,
        filter=None if prefixes is None else ",".join(prefixes),
        strict=strict,
    )
    return BufferRegistry(
        budget_bytes=budget_bytes, accepts=config.accepts, strict=strict
    )


class TestCudaGraphDumpConfig:
    def test_accepts_everything_without_filter(self):
        config = CudaGraphDumpConfig(enable=True)
        assert config.prefixes == ()
        assert config.accepts("anything")

    def test_prefixes_are_split_and_stripped(self):
        config = CudaGraphDumpConfig(enable=True, filter=" a.b , ,c ")
        assert config.prefixes == ("a.b", "c")
        assert config.accepts("a.b.qkv_proj")
        assert config.accepts("c")
        assert not config.accepts("b.a")

    def test_budget_bytes(self):
        assert CudaGraphDumpConfig(budget_mb=2).budget_bytes == 2 * 1024 * 1024
        assert CudaGraphDumpConfig(budget_mb=-1).budget_bytes == -1

    def test_from_dumper_config_reads_prefixed_fields(self):
        dumper_config = SimpleNamespace(
            cuda_graph_enable=True,
            cuda_graph_filter="x",
            cuda_graph_budget_mb=7,
            cuda_graph_strict=True,
        )
        config = CudaGraphDumpConfig.from_dumper_config(dumper_config)
        assert (config.enable, config.filter, config.budget_mb, config.strict) == (
            True,
            "x",
            7,
            True,
        )

    def test_from_dumper_config_tolerates_missing_fields(self):
        config = CudaGraphDumpConfig.from_dumper_config(SimpleNamespace())
        assert not config.enable


class TestBufferKey:
    def test_str_hides_the_zeroth_occurrence(self):
        assert str(BufferKey("a.b")) == "a.b"
        assert str(BufferKey("a.b", 2)) == "a.b#2"

    def test_occurrences_are_distinct_keys(self):
        assert BufferKey("a", 0) != BufferKey("a", 1)
        assert len({BufferKey("a", 0), BufferKey("a", 0)}) == 1


class TestBufferRegistry:
    def test_record_returns_a_flat_slice_of_the_right_length(self):
        registry = _registry()
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        slot = registry.record(BufferKey("a"), "shape0", tensor)
        assert slot is not None
        assert slot.shape == (6,)
        assert registry.num_buffers == 1

    def test_occurrences_get_separate_buffers(self):
        registry = _registry()
        tensor = torch.zeros(4)
        first = registry.record(BufferKey("a", 0), "shape0", tensor)
        second = registry.record(BufferKey("a", 1), "shape0", tensor)
        assert first.data_ptr() != second.data_ptr()
        assert registry.num_buffers == 2

    def test_same_key_reuses_one_buffer_across_shapes(self):
        registry = _registry()
        key = BufferKey("a")
        big = registry.record(key, "big", torch.zeros(8))
        small = registry.record(key, "small", torch.zeros(4))
        assert big.data_ptr() == small.data_ptr()
        assert registry.num_buffers == 1

    def test_filtered_names_cost_no_memory(self):
        registry = _registry(prefixes=["keep"])
        assert (
            registry.record(BufferKey("drop.me"), "shape0", torch.zeros(1024)) is None
        )
        assert registry.used_bytes == 0
        assert registry.rejected[BufferKey("drop.me")] == "filtered"
        assert (
            registry.record(BufferKey("keep.me"), "shape0", torch.zeros(4)) is not None
        )


class TestBufferRegistryLimits:
    def test_budget_rejects_and_warns_by_default(self):
        registry = _registry(budget_bytes=16)
        assert registry.record(BufferKey("a"), "s", torch.zeros(4)) is not None
        assert registry.record(BufferKey("b"), "s", torch.zeros(4)) is None
        assert registry.rejected[BufferKey("b")] == "budget"
        # A rejected key stays rejected without re-running the budget check.
        assert registry.record(BufferKey("b"), "s", torch.zeros(1)) is None

    def test_budget_raises_in_strict_mode(self):
        registry = _registry(budget_bytes=0, strict=True)
        with pytest.raises(DumpBudgetExceeded):
            registry.record(BufferKey("a"), "s", torch.zeros(4))

    def test_unlimited_budget(self):
        registry = _registry(budget_bytes=-1)
        assert registry.record(BufferKey("a"), "s", torch.zeros(1024)) is not None

    def test_growth_is_rejected_not_truncated(self):
        registry = _registry()
        registry.record(BufferKey("a"), "small", torch.zeros(4))
        with pytest.raises(DumpBufferGrowthRejected):
            registry.record(BufferKey("a"), "big", torch.zeros(8))

    def test_dtype_change_is_rejected(self):
        registry = _registry()
        registry.record(BufferKey("a"), "s", torch.zeros(4, dtype=torch.float32))
        with pytest.raises(DumpBufferGrowthRejected):
            registry.record(BufferKey("a"), "s", torch.zeros(4, dtype=torch.float16))

    def test_addresses_stay_stable(self):
        registry = _registry()
        registry.record(BufferKey("a"), "s", torch.zeros(4))
        registry.assert_addresses_stable()


class TestBufferRegistryViews:
    def test_views_reshape_per_shape_token(self):
        registry = _registry()
        key = BufferKey("a")
        registry.record(key, "big", torch.zeros(2, 4)).copy_(torch.arange(8.0))
        registry.record(key, "small", torch.zeros(2, 2))
        big = dict(registry.views("big"))
        small = dict(registry.views("small"))
        assert big[key].shape == (2, 4)
        assert small[key].shape == (2, 2)
        # The small view aliases the head of the same flat buffer.
        assert torch.equal(small[key].reshape(-1), torch.arange(4.0))

    def test_views_skip_keys_never_seen_under_that_token(self):
        registry = _registry()
        registry.record(BufferKey("a"), "s0", torch.zeros(4))
        registry.record(BufferKey("b"), "s1", torch.zeros(4))
        assert [key.name for key, _ in registry.views("s0")] == ["a"]
        assert [key.name for key, _ in registry.views("s1")] == ["b"]
        assert list(registry.views("s2")) == []

    def test_names(self):
        registry = _registry()
        registry.record(BufferKey("a", 0), "s", torch.zeros(1))
        registry.record(BufferKey("a", 1), "s", torch.zeros(1))
        assert registry.names() == {"a"}


class TestTagInterning:
    def test_round_trip_and_stability(self):
        registry = _registry()
        key = BufferKey("a", 1)
        tag = registry.tag_for(key)
        assert registry.tag_for(key) == tag
        assert registry.key_for_tag(tag) == key

    def test_distinct_keys_get_distinct_tags(self):
        registry = _registry()
        assert registry.tag_for(BufferKey("a")) != registry.tag_for(BufferKey("b"))


class TestRowPairs:
    def test_plain_decode_offers_one_pair(self):
        runner = SimpleNamespace(raw_num_token=3, raw_bs=3, bs=8)
        assert _row_pairs(runner, _ShapeToken(size=8)) == [(8, 3)]

    def test_spec_decode_offers_token_and_request_bases(self):
        # bs=32 padded requests x width=4 -> 128 padded token rows; 4 real
        # requests -> 16 real token rows. A 32-row per-request tensor must trim
        # to 4, not to 16, which is what a single (size, raw_num_token) pair
        # would have done.
        runner = SimpleNamespace(
            raw_num_token=16, raw_bs=4, bs=32, captured_req_width=4
        )
        pairs = _row_pairs(runner, _ShapeToken(size=128))
        assert pairs == [(128, 16), (32, 4)]
        assert _trim(torch.zeros(32, 5), pairs).shape == (4, 5)
        assert _trim(torch.zeros(128, 5), pairs).shape == (16, 5)

    def test_ragged_verify_keys_on_num_tokens(self):
        runner = SimpleNamespace(
            raw_num_token=5,
            raw_bs=2,
            bs=4,
            captured_req_width=4,
            ragged_verify_mode=True,
        )
        assert _row_pairs(runner, _ShapeToken(size=16)) == [(16, 5), (4, 2)]

    def test_prefill_style_runner_keys_on_num_tokens(self):
        runner = SimpleNamespace(raw_num_tokens=5, bs=None)
        assert _row_pairs(runner, _ShapeToken(size=16)) == [(16, 5)]

    def test_unpadded_batch_offers_nothing(self):
        runner = SimpleNamespace(raw_num_token=8, raw_bs=8, bs=8)
        assert _row_pairs(runner, _ShapeToken(size=8)) == []


class TestTrim:
    def test_only_an_exact_dim0_match_trims(self):
        pairs = [(8, 3)]
        assert _trim(torch.zeros(8, 2), pairs).shape == (3, 2)
        assert _trim(torch.zeros(7, 2), pairs).shape == (7, 2)
        assert _trim(torch.zeros(2, 8), pairs).shape == (2, 8)

    def test_scalars_pass_through(self):
        assert _trim(torch.tensor(1.0), [(8, 3)]).dim() == 0

    def test_no_pairs_is_a_no_op(self):
        assert _trim(torch.zeros(8), []).shape == (8,)


def _state(enable: bool = True, **config_kwargs) -> _CudaGraphDumpState:
    """A state object with the registry injected.

    Bypasses `_ensure_configured`, which would read the real dumper's config and
    register an `atexit` hook -- neither of which belongs in a unit test.
    """
    state = _CudaGraphDumpState()
    state._config = CudaGraphDumpConfig(enable=enable, **config_kwargs)
    state._registry = BufferRegistry(
        budget_bytes=1 << 30,
        accepts=state._config.accepts,
        strict=state._config.strict,
    )
    return state


@contextmanager
def _pretend_stream_is_capturing(monkeypatch):
    monkeypatch.setattr(state_module, "_stream_is_capturing", lambda: True)
    yield


class _FakeDumper:
    """Stands in for the module-level `dumper` singleton at collect time."""

    def __init__(self):
        self.frames = []

    def dump(self, name, value, **tags):
        self.frames.append((name, value, tags))


class TestTapLadder:
    def test_disabled_hands_the_tensor_back(self):
        assert _state(enable=False).tap("a", torch.zeros(4)) is False

    def test_non_tensors_are_never_taken(self):
        assert _state().tap("a", "not a tensor") is False
        assert _state().tap("a", 3) is False

    def test_idle_is_the_eager_path(self):
        # No scope open: genuine eager execution, the caller writes the file.
        assert _state().tap("a", torch.zeros(4)) is False

    def test_setup_frames_are_swallowed(self):
        state = _state()
        with state.setup_scope():
            # Warmup / compile-pass dummies: taken (True) but not recorded.
            assert state.tap("a", torch.zeros(4)) is True
        assert state.registry.num_buffers == 0

    def test_capture_without_a_capturing_stream_is_swallowed(self):
        state = _state()
        with state.capture_scope():
            assert state.tap("a", torch.zeros(4)) is True
        assert state.registry.num_buffers == 0


class TestEagerTags:
    def test_disabled_adds_nothing(self):
        # A default run's frames must be byte-for-byte what they always were.
        assert _state(enable=False).eager_tags() == {}

    def test_enabled_labels_the_eager_channel(self):
        # Without this the acceptance gate cannot tell a declined frame from a
        # pure-eager baseline frame, and "t1|t3 plus eager equals eager-only"
        # is unverifiable.
        assert _state().eager_tags() == {"tap": "eager"}


class TestOccurrenceCounter:
    def test_repeated_names_get_distinct_occurrences(self):
        state = _state()
        assert [state._next_key("a").occurrence for _ in range(3)] == [0, 1, 2]
        assert state._next_key("b").occurrence == 0

    def test_begin_forward_resets(self):
        state = _state()
        state._next_key("a")
        state.begin_forward()
        assert state._next_key("a").occurrence == 0

    def test_wrap_forward_fn_resets_per_invocation(self):
        state = _state()
        seen = []

        def forward_fn():
            seen.append(state._next_key("a").occurrence)

        wrapped = state.wrap_forward_fn(forward_fn)
        # capture_one calls forward_fn twice; the second pass must not allocate a
        # second buffer for every name.
        wrapped()
        wrapped()
        assert seen == [0, 0]


class TestT1EndToEnd:
    def _record_one_shape(self, state, token, rows):
        with state.capture_scope(), state.shape_scope(token):
            assert state.tap(
                "layer.qkv_proj", torch.arange(rows * 2.0).reshape(rows, 2)
            )
            assert state.tap("layer.o_proj", torch.zeros(rows, 2))

    def test_capture_records_and_collect_writes(self, monkeypatch):
        state = _state()
        fake = _FakeDumper()
        monkeypatch.setattr("sglang.srt.debug_utils.dumper.dumper", fake)
        token = _ShapeToken(size=8)
        with _pretend_stream_is_capturing(monkeypatch):
            self._record_one_shape(state, token, rows=8)
        assert state.registry.num_buffers == 2

        runner = SimpleNamespace(raw_num_token=3, raw_bs=3, bs=8)
        with state.runner_scope(runner, graph_runner="decode", pd_role=None):
            written = state.collect(token, graph_backend="full", graph_size=8)
        assert written == 2
        assert {name for name, _, _ in fake.frames} == {
            "layer.qkv_proj",
            "layer.o_proj",
        }
        tags = fake.frames[0][2]
        assert tags["tap"] == "t1"
        assert tags["graph_backend"] == "full"
        assert tags["graph_runner"] == "decode"
        # pd_role was None, so runner_scope dropped it rather than tagging a None.
        assert "pd_role" not in tags
        # 8 padded rows trimmed to the 3 real ones.
        assert all(value.shape == (3, 2) for _, value, _ in fake.frames)

    def test_second_occurrence_is_tagged_and_kept_apart(self, monkeypatch):
        state = _state()
        fake = _FakeDumper()
        monkeypatch.setattr("sglang.srt.debug_utils.dumper.dumper", fake)
        token = _ShapeToken(size=4)
        with _pretend_stream_is_capturing(monkeypatch):
            with state.capture_scope(), state.shape_scope(token):
                state.tap("mlp.down_proj", torch.zeros(4, 2))
                state.tap("mlp.down_proj", torch.ones(4, 2))
        assert state.registry.num_buffers == 2

        with state.runner_scope(SimpleNamespace(raw_num_token=4, raw_bs=4, bs=4)):
            assert state.collect(token) == 2
        occurrences = sorted(
            (tags.get("occurrence") for _, _, tags in fake.frames),
            key=lambda occurrence: -1 if occurrence is None else occurrence,
        )
        # The zeroth occurrence is left untagged so the common case stays clean.
        assert occurrences == [None, 1]
        assert all(name == "mlp.down_proj" for name, _, _ in fake.frames)

    def test_collect_only_sees_the_requested_shape(self, monkeypatch):
        state = _state()
        fake = _FakeDumper()
        monkeypatch.setattr("sglang.srt.debug_utils.dumper.dumper", fake)
        big, small = _ShapeToken(size=8), _ShapeToken(size=4)
        with _pretend_stream_is_capturing(monkeypatch):
            self._record_one_shape(state, big, rows=8)
            state.begin_forward()
            self._record_one_shape(state, small, rows=4)
        # One buffer per name, shared across both shapes.
        assert state.registry.num_buffers == 2
        with state.runner_scope(SimpleNamespace(bs=4)):
            assert state.collect(small) == 2
        assert all(value.shape == (4, 2) for _, value, _ in fake.frames)

    def test_disabled_collect_writes_nothing(self):
        assert _state(enable=False).collect(_ShapeToken(size=8)) == 0


class TestReport:
    def test_zero_taps_warns_once(self, caplog):
        state = _state()
        state.report()
        state.report()
        assert sum("no tap fired" in r.getMessage() for r in caplog.records) == 1

    def test_zero_taps_raises_in_strict_mode(self):
        with pytest.raises(RuntimeError, match="no tap fired"):
            _state(strict=True).report()

    def test_at_exit_swallows_the_strict_failure(self, caplog):
        # atexit must not turn a diagnostic into interpreter shutdown noise.
        _state(strict=True)._report_at_exit()
        assert any("report failed" in r.getMessage() for r in caplog.records)


class TestSeamHelpers:
    def test_first_arg_reads_positional_and_keyword_forms(self):
        assert _first_arg(("a",), {}, "shape_key") == "a"
        assert _first_arg((), {"shape_key": "a"}, "shape_key") == "a"
        assert _first_arg((), {}, "shape_key") is None

    def test_backend_label_falls_back_to_the_class_name(self):
        assert _backend_label(type("FullCudaGraphBackend", (), {})()) == "full"
        assert (
            _backend_label(type("TcPiecewiseCudaGraphBackend", (), {})())
            == "tc_piecewise"
        )
        assert _backend_label(type("NpuFooBackend", (), {})()) == "NpuFooBackend"

    def test_pd_role_is_none_without_a_published_context(self):
        # Co-located, or a unit test with no runtime context: must not raise.
        assert _pd_role() in (None, "prefill", "decode")

    def test_wrap_forward_fn_handles_both_call_shapes(self, monkeypatch):
        state = _state()
        monkeypatch.setattr(seams_module, "cuda_graph_dump", state)
        calls = []
        args, kwargs = _wrap_forward_fn(("key", lambda: calls.append("pos")), {})
        args[1]()
        args, kwargs = _wrap_forward_fn(
            ("key",), {"forward_fn": lambda: calls.append("kw")}
        )
        kwargs["forward_fn"]()
        assert calls == ["pos", "kw"]
        # Nothing to substitute: passed through untouched.
        assert _wrap_forward_fn(("key",), {}) == (("key",), {})

    def test_wrap_is_idempotent_and_dict_local(self):
        class Base:
            def execute(self):
                return "base"

        class Child(Base):
            pass

        seen = []

        def factory(fn):
            def wrapped(self):
                seen.append(1)
                return fn(self)

            return wrapped

        _wrap(Base, "execute", factory)
        _wrap(Base, "execute", factory)  # _SEAM_MARK short-circuits the second
        _wrap(Child, "execute", factory)  # no override in __dict__: nothing to wrap
        assert "execute" not in Child.__dict__
        Child().execute()
        assert seen == [1]


class _DummyBackend:
    """The shape of a BaseCudaGraphBackend subclass, without the import weight.

    `runner_backend/__init__.py` eagerly imports all three real backends, which
    would drag the whole model executor into a CPU test. The seam only ever
    touches these five methods, so a stand-in exercises it faithfully.
    """

    def __init__(self):
        self.replayed = []

    @contextmanager
    def capture_session(self, stream):
        yield "session"

    def capture_one(self, shape_key, forward_fn, **kwargs):
        forward_fn()
        forward_fn()

    def replay(self, shape_key, static_forward_batch, **kwargs):
        self.replayed.append(shape_key)
        return "logits"

    def cleanup(self):
        return None


def _fresh_backend() -> type:
    """A new class whose *own* `__dict__` holds every seam target.

    `_wrap` only ever replaces entries in `cls.__dict__`, which mirrors the real
    backends: each one is a concrete override of the ABC. A plain subclass of
    `_DummyBackend` would inherit the methods and get nothing wrapped.
    """
    body = {
        k: v
        for k, v in _DummyBackend.__dict__.items()
        if k not in ("__dict__", "__weakref__")
    }
    return type("Backend", (), body)


class TestSeamIntegration:
    def _armed(self, monkeypatch):
        state = _state()
        monkeypatch.setattr(seams_module, "cuda_graph_dump", state)
        cls = _fresh_backend()
        install_backend_seam(cls)
        install_backend_seam(cls)  # idempotent
        return state, cls

    def test_capture_then_replay_writes_one_frame_per_buffer(self, monkeypatch):
        state, Backend = self._armed(monkeypatch)
        fake = _FakeDumper()
        monkeypatch.setattr("sglang.srt.debug_utils.dumper.dumper", fake)
        token = _ShapeToken(size=4)

        backend = Backend()
        with _pretend_stream_is_capturing(monkeypatch):
            with backend.capture_session(stream=None) as session:
                assert session == "session"
                backend.capture_one(token, lambda: state.tap("a.b", torch.zeros(4, 2)))
        # forward_fn ran twice; wrap_forward_fn reset the counter, so one buffer.
        assert state.registry.num_buffers == 1

        runner = SimpleNamespace(raw_num_token=2, raw_bs=2, bs=4)
        with state.runner_scope(runner):
            assert backend.replay(token, None) == "logits"
        assert [name for name, _, _ in fake.frames] == ["a.b"]
        tags = fake.frames[0][2]
        assert tags["tap"] == "t1"
        assert tags["graph_backend"] == "Backend"
        assert tags["graph_size"] == 4
        assert fake.frames[0][1].shape == (2, 2)

    def test_capture_outside_a_session_records_nothing(self, monkeypatch):
        state, Backend = self._armed(monkeypatch)
        with _pretend_stream_is_capturing(monkeypatch):
            # No capture_session open: the ladder must not reach the T1 rung.
            Backend().capture_one(
                _ShapeToken(size=4), lambda: state.tap("a.b", torch.zeros(4))
            )
        assert state.registry.num_buffers == 0

    def test_cleanup_reports(self, monkeypatch):
        state, Backend = self._armed(monkeypatch)
        state._config = CudaGraphDumpConfig(enable=True, strict=True)
        with pytest.raises(RuntimeError, match="no tap fired"):
            Backend().cleanup()

    def test_runner_seam_publishes_the_runner_and_resets_occurrences(self, monkeypatch):
        state = _state()
        monkeypatch.setattr(seams_module, "cuda_graph_dump", state)
        seen = []

        class Runner:
            bs = 8

            def execute(self, forward_batch):
                seen.append(
                    (
                        state._runner_ctx.get("runner") is self,
                        state._next_key("a").occurrence,
                    )
                )
                return "out"

        install_runner_seam(Runner)
        runner = Runner()
        batch = SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE"))
        assert runner.execute(batch) == "out"
        assert runner.execute(batch) == "out"
        # Published both times, and begin_forward() zeroed the counter each time.
        assert seen == [(True, 0), (True, 0)]
        # The scope closed: nothing leaks into the next eager forward.
        assert state._runner_ctx == {}
