"""CPU tests for the order of snapshot loading (weight_cache/snapshot_loader.py)."""

import ast
import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch
from torch import nn
from transformers import PretrainedConfig

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.weight_cache import load_diff, schema, snapshot_loader, validator
from sglang.srt.weight_cache.config import build_load_spec
from sglang.srt.weight_cache.coordinator import Ballot
from sglang.srt.weight_cache.schema import LoadSpec, SnapshotManifest, StorageMeta
from sglang.srt.weight_cache.transport import WeightTransport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CPU = DeviceConfig("cpu")


def _model_config(hf_config=None, model_path="m"):
    return SimpleNamespace(
        hf_config=PretrainedConfig() if hf_config is None else hf_config,
        model_path=model_path,
        revision=None,
        quantization=None,
        dtype=torch.float32,
    )


@contextlib.contextmanager
def _topology(tp_rank=0, tp_size=2):
    with (
        get_parallel().override(
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_tp_rank=tp_rank,
            moe_tp_size=tp_size,
            moe_tp_rank=tp_rank,
            pp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
        ),
        mock.patch("sglang.srt.layers.dp_attention.get_moe_cp_size", return_value=1),
    ):
        yield


def _spec(tp_rank, tp_size=2, *, hf_config=None, model_path="m", draft_model_idx=None):
    with _topology(tp_rank, tp_size):
        return build_load_spec(
            checkpoint_id="rev1",
            model_config=_model_config(hf_config, model_path),
            load_config=LoadConfig(draft_model_idx=draft_model_idx),
            target_device=_CPU.device,
        )


def _pre_load(loader, model_config, *, checkpoint_id="rev1"):
    return snapshot_loader.pre_load_check(
        loader=loader,
        model_config=model_config,
        checkpoint_id=checkpoint_id,
        device_config=_CPU,
        group=None,
    )


class _QuantMethod:
    processed = False

    def process_weights_after_loading(self, layer):
        self.processed = True


class _TinyModel(nn.Module):
    def __init__(self, config, quant_config=None, **kwargs):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.proj.quant_method = _QuantMethod()
        self.post_processed = False

    def post_load_weights(self):
        self.post_processed = True


class _MemoryTransport(WeightTransport):
    """Copies bytes in memory; publish consumes its input, as vmm_fd does."""

    def __init__(self, fetch_error=None):
        self.published, self.closed, self.fetch_error = {}, False, fetch_error

    def publish(self, *, storages):
        self.published = {name: t.clone() for name, t in storages.items()}
        storages.clear()
        return {"kind": "memory"}

    def fetch(self, *, manifest, device):
        if self.fetch_error is not None:
            raise self.fetch_error
        return {name: self.published[name].clone() for name in manifest.storages}

    def close(self):
        self.closed = True


class _Source(snapshot_loader.SnapshotLoader):
    def __init__(self, load_config, found=None, owner=False):
        super().__init__(load_config, owner=owner)
        self.found, self.announced, self.withdrawn = found, [], []

    def download_model(self, model_config):
        pass

    def _find(self, *, spec, device_config):
        return self.found

    def _announce(self, *, spec, manifest):
        self.announced.append(manifest)

    def _withdraw(self, *, spec, manifest):
        self.withdrawn.append(manifest)


class _RawLoader:
    def __init__(self, load_config):
        self.load_config = load_config

    def load_model(self, *, model_config, device_config):
        # Presharded builds a throwaway model for its signature before the real one.
        self.throwaway = loader_mod._initialize_model(model_config, self.load_config)
        model = loader_mod._initialize_model(model_config, self.load_config)
        model.post_load_weights()  # stands in for loading weights
        return model


def _vote_failing_at(failing_call=None):
    """Two ranks; raises on an error ballot, or for a peer at ``failing_call``."""
    calls = []

    def vote(*, group, ballot):
        calls.append(ballot)
        if ballot.error is not None:
            raise RuntimeError(ballot.error)
        if len(calls) == failing_call:
            raise RuntimeError(f"rank 1 failed at vote {failing_call}")
        return [ballot, ballot]

    vote.calls = calls
    return vote


def _imported_names(module) -> set:
    names = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module)
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return {name for name in names if name.startswith("sglang")}


class TestSnapshotLoader(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        patcher = mock.patch.object(
            loader_mod, "get_model_architecture", return_value=(_TinyModel, "Tiny")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @contextlib.contextmanager
    def _owner_parts(self, **fakes):
        parts = {
            "export": lambda *, model, constructed: (
                {"proj": model.proj.weight.data.view(torch.uint8)},
                b"tables",
            ),
            "attest": lambda *, model, storages, constructed: b"attestation",
            "vote": _vote_failing_at(),
            **fakes,
        }
        with (
            mock.patch.multiple(
                load_diff,
                record_construction=lambda m: ("diff", id(m), m.post_processed),
                export=parts["export"],
            ),
            mock.patch.multiple(
                validator,
                record_construction=lambda m: ("check", id(m), m.post_processed),
                attest=parts["attest"],
            ),
            mock.patch.multiple(
                snapshot_loader,
                get_model_loader=mock.Mock(side_effect=lambda cfg, _: _RawLoader(cfg)),
                vote=parts["vote"],
            ),
        ):
            yield

    def _publish(self, source, transport, model_config=None):
        if model_config is None:
            model_config = _model_config()
        # Owner failure cases count the two publish votes independently of lookup.
        with (
            _topology(),
            mock.patch.object(snapshot_loader, "vote", _vote_failing_at()),
        ):
            self.assertIsNone(_pre_load(source, model_config))
        model = source.load_model(model_config=model_config, device_config=_CPU)
        exported = source.export_snapshot(model=model)
        manifest = source.publish_snapshot(
            exported=exported, transport=transport, group=None
        )
        return model, manifest

    def test_owner_records_construction_of_the_model_it_publishes(self):
        seen, transport = {}, _MemoryTransport()

        def attest(*, model, storages, constructed):
            seen["attest"] = (sorted(storages), constructed)
            return b"attestation"

        source = _Source(
            LoadConfig(fallback_load_format=LoadFormat.SAFETENSORS, draft_model_idx=1),
            owner=True,
        )
        with self._owner_parts(attest=attest):
            model, manifest = self._publish(source, transport)
            raw_config = snapshot_loader.get_model_loader.call_args.args[0]

        # The raw load keeps the engine's load config, with only the format changed.
        self.assertEqual(raw_config.load_format, LoadFormat.SAFETENSORS)
        self.assertEqual(raw_config.draft_model_idx, 1)
        # Recorded before the weights loaded, for the returned model, not the throwaway.
        self.assertEqual(seen["attest"], (["proj"], ("check", id(model), False)))
        self.assertFalse(model.training)
        self.assertEqual(transport.published["proj"].numel(), 16)
        self.assertEqual(
            manifest.storages, {"proj": StorageMeta(nbytes=16, device="cpu")}
        )
        self.assertEqual(source.announced, [manifest])

    def test_reader_finds_owner_snapshot_under_the_pre_load_spec(self):
        owner = _Source(LoadConfig(), owner=True)
        reader = _Source(LoadConfig())
        transport = _MemoryTransport()
        model_config = _model_config()
        registry = {}

        def announce(*, spec, manifest):
            registry[spec.digest()] = (manifest, transport)

        def find(*, spec, device_config):
            return registry.get(spec.digest())

        with (
            self._owner_parts(),
            mock.patch.object(owner, "_announce", announce),
            mock.patch.object(reader, "_find", find),
            mock.patch.object(
                _TinyModel,
                "post_load_weights",
                side_effect=lambda: setattr(model_config.hf_config, "hidden_size", 128),
            ),
            mock.patch.object(
                snapshot_loader, "build_load_spec", wraps=build_load_spec
            ) as build,
        ):
            _, manifest = self._publish(owner, transport, model_config)
            self.assertEqual(model_config.hf_config.hidden_size, 128)
            with _topology():
                self.assertIs(_pre_load(reader, _model_config()), manifest)
            self.assertEqual(build.call_count, 2)

    def test_owner_rejects_an_export_from_a_previous_lookup(self):
        owner = _Source(LoadConfig(), owner=True)
        model_config = _model_config()
        transport = _MemoryTransport()
        vote = _vote_failing_at()
        with (
            self._owner_parts(vote=vote),
            _topology(),
            mock.patch.object(transport, "publish", wraps=transport.publish) as publish,
        ):
            _pre_load(owner, model_config, checkpoint_id="first")
            model = owner.load_model(model_config=model_config, device_config=_CPU)
            exported = owner.export_snapshot(model=model)
            self.assertIsNone(exported.error)

            _pre_load(owner, model_config, checkpoint_id="second")
            self.assertEqual(len(vote.calls), 2)
            with self.assertRaises(RuntimeError):
                owner.publish_snapshot(
                    exported=exported, transport=transport, group=None
                )
        self.assertEqual(len(vote.calls), 3)
        self.assertIsNotNone(vote.calls[-1].error)
        publish.assert_not_called()
        self.assertTrue(transport.closed)
        self.assertFalse(owner.announced)

    def test_every_rank_mints_the_version_from_all_attestations(self):
        gathered = [Ballot(payload=b"rank0"), Ballot(payload=b"rank1")]
        changed = [Ballot(payload=b"rank0"), Ballot(payload=b"changed")]
        versions = []
        for attestation, ballots in (
            (b"a0", gathered),
            (b"a1", gathered),
            (b"a0", changed),
        ):
            with self._owner_parts(
                attest=lambda **kwargs: attestation,
                vote=lambda *, group, ballot: ballots,
            ):
                _, manifest = self._publish(
                    _Source(LoadConfig(), owner=True), _MemoryTransport()
                )
            versions.append(manifest.version)
        self.assertEqual(versions[0], versions[1])
        self.assertNotEqual(versions[0], versions[2])

    def test_owner_failure_on_any_rank_leaves_nothing_findable(self):
        def export(*, model, constructed):
            raise ValueError("unsupported tensor")

        cases = {
            "own export fails": (dict(export=export), "unsupported tensor", 1),
            "peer fails to publish": (dict(vote=_vote_failing_at(1)), "vote 1", 1),
            "peer fails to announce": (dict(vote=_vote_failing_at(2)), "vote 2", 2),
        }
        for name, (fakes, message, votes) in cases.items():
            with self.subTest(name):
                source, transport = (
                    _Source(LoadConfig(), owner=True),
                    _MemoryTransport(),
                )
                fakes.setdefault("vote", _vote_failing_at())
                with self._owner_parts(**fakes):
                    with self.assertRaisesRegex(RuntimeError, message):
                        self._publish(source, transport)
                # Every rank votes, even one that failed, so no peer waits it out.
                self.assertEqual(len(fakes["vote"].calls), votes)
                self.assertTrue(transport.closed)
                self.assertEqual(source.withdrawn, source.announced)

    def test_reader_raw_load_records_nothing(self):
        def record(model):
            raise AssertionError("only an owner records construction")

        with (
            self._owner_parts(),
            mock.patch.object(validator, "record_construction", record),
            mock.patch.object(load_diff, "record_construction", record),
        ):
            model = _Source(LoadConfig()).load_model(
                model_config=_model_config(), device_config=_CPU
            )
        self.assertFalse(model.training)

    def _read(self, *, fetch_error=None, install_error=None, validate_error=None):
        manifest = SnapshotManifest(
            version="v1",
            storages={"proj": StorageMeta(nbytes=16, device="cpu")},
            load_diff=b"tables",
            attestation=b"attestation",
            source={},
        )
        self.transport = _MemoryTransport(fetch_error=fetch_error)
        self.transport.published = {"proj": torch.zeros(16, dtype=torch.uint8)}
        source = _Source(LoadConfig(), found=(manifest, self.transport))
        model_config = _model_config()
        with (
            _topology(),
            mock.patch.object(snapshot_loader, "vote", _vote_failing_at()),
        ):
            self.assertIs(_pre_load(source, model_config), manifest)

        def install(*, skeleton, manifest, storages):
            if install_error is not None:
                raise install_error
            self.skeleton = skeleton
            return _TinyModel(None)

        def validate(*, model, storages, attestation, constructed):
            if validate_error is not None:
                raise validate_error
            # Recorded on the meta skeleton; compared in eval mode, as the owner is.
            self.assertEqual(constructed, ("check", True))
            self.assertFalse(model.training)

        with (
            mock.patch.multiple(
                validator,
                record_construction=lambda m: ("check", m.proj.weight.is_meta),
                validate=validate,
            ),
            mock.patch.object(load_diff, "install", install),
        ):
            return source.load_model(model_config=model_config, device_config=_CPU)

    def test_reader_installs_without_post_processing(self):
        model = self._read()
        for module in (self.skeleton, model):
            self.assertFalse(module.post_processed)
            self.assertFalse(module.proj.quant_method.processed)
        self.assertFalse(self.transport.closed)  # the bytes live as long as the model

    def test_reader_failure_after_lookup_unmaps(self):
        # vmm_fd maps the whole snapshot in fetch, and nothing falls back after it.
        for step in ("fetch", "install", "validate"):
            with self.subTest(step):
                with self.assertRaisesRegex(RuntimeError, step):
                    self._read(**{f"{step}_error": RuntimeError(step)})
                self.assertTrue(self.transport.closed)

    def test_load_spec_separates_shard_identity_from_shared_settings(self):
        spec = _spec(0)
        self.assertEqual(
            msgspec.json.decode(msgspec.json.encode(spec), type=LoadSpec), spec
        )
        unknown_field = {**msgspec.to_builtins(spec), "future_layout": "new"}
        with self.assertRaisesRegex(msgspec.ValidationError, "future_layout"):
            msgspec.json.decode(msgspec.json.encode(unknown_field), type=LoadSpec)
        for field in ("tp_rank", "pp_rank", "moe_dp_rank", "moe_ep_rank"):
            with self.subTest(field=field):
                peer = msgspec.structs.replace(spec, **{field: 1})
                self.assertNotEqual(spec.digest(), peer.digest())
                self.assertEqual(
                    spec.digest(include_rank=False), peer.digest(include_rank=False)
                )
        staged = _spec(
            0,
            hf_config=PretrainedConfig(name_or_path="/tmp/staging"),
            model_path="/tmp/staging",
        )
        self.assertEqual(spec.digest(), staged.digest())
        self.assertEqual(
            spec.digest(include_rank=False), staged.digest(include_rank=False)
        )
        for other in (
            _spec(0, tp_size=4),
            _spec(0, draft_model_idx=1),
            _spec(0, hf_config=PretrainedConfig(hidden_size=128)),
            msgspec.structs.replace(spec, checkpoint_id="rev2"),
            msgspec.structs.replace(spec, flashinfer_cutedsl_swap_ab=True),
            msgspec.structs.replace(spec, fp4_gemm_runner_backend="flashinfer_trtllm"),
        ):
            self.assertNotEqual(spec.digest(), other.digest())
            self.assertNotEqual(
                spec.digest(include_rank=False), other.digest(include_rank=False)
            )

    def test_pre_load_check_votes_on_every_outcome_but_the_rank(self):
        ballots = []
        manifest = SnapshotManifest(
            version="v1", storages={}, load_diff=b"", attestation=b"", source={}
        )
        failing = _Source(LoadConfig())
        failing._find = mock.Mock(side_effect=OSError("daemon socket refused"))
        found = _Source(LoadConfig(), found=(manifest, _MemoryTransport()))
        cases = [
            (_Source(LoadConfig()), 0, 2),
            (_Source(LoadConfig()), 1, 2),
            (failing, 0, 2),
            (found, 1, 2),
            (_Source(LoadConfig()), 0, 4),
        ]
        with mock.patch.object(
            snapshot_loader, "vote", lambda *, group, ballot: ballots.append(ballot)
        ):
            for loader, rank, size in cases:
                with _topology(rank, size):
                    _pre_load(loader, _model_config())
        self.assertEqual(ballots[0].agree, ballots[1].agree)
        self.assertIn("daemon socket refused", ballots[2].error)
        self.assertNotEqual(ballots[3].agree, ballots[0].agree)
        self.assertNotEqual(ballots[4].agree, ballots[0].agree)

        # When the ranks reject the load, a rank that found a snapshot releases it.
        transport = _MemoryTransport()
        rejected = _Source(LoadConfig(), found=(manifest, transport))
        with (
            _topology(),
            mock.patch.object(snapshot_loader, "vote", _vote_failing_at(1)),
        ):
            with self.assertRaises(RuntimeError):
                _pre_load(rejected, _model_config())
        self.assertTrue(transport.closed)

    def test_pre_load_check_votes_on_build_failure_before_lookup(self):
        source = _Source(LoadConfig())
        vote = _vote_failing_at()
        with (
            _topology(),
            mock.patch.object(source, "_find") as find,
            mock.patch.object(snapshot_loader, "vote", vote),
            self.assertRaisesRegex(RuntimeError, "checkpoint"),
        ):
            _pre_load(source, _model_config(), checkpoint_id="")
        find.assert_not_called()
        self.assertEqual(len(vote.calls), 1)
        self.assertIn("checkpoint", vote.calls[0].error)

    def test_load_diff_and_validator_share_only_the_schema(self):
        # A validator built on load-diff's logic would only check what load-diff did.
        package = "sglang.srt.weight_cache"
        diff_imports = _imported_names(load_diff)
        check_imports = _imported_names(validator)
        shared = (diff_imports & check_imports) - {package}
        self.assertTrue(all(name.startswith(f"{package}.schema") for name in shared))
        self.assertFalse([n for n in check_imports if "load_diff" in n])
        self.assertFalse([n for n in diff_imports if "validator" in n])
        self.assertFalse(_imported_names(schema))


if __name__ == "__main__":
    unittest.main()
