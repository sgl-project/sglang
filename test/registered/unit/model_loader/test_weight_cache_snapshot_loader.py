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
from sglang.srt.weight_cache.schema import SnapshotManifest, StorageMeta
from sglang.srt.weight_cache.transport import WeightTransport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CPU = DeviceConfig("cpu")
_GROUP = SimpleNamespace(rank_in_group=0)


def _model_config():
    return SimpleNamespace(
        hf_config=PretrainedConfig(),
        model_path="m",
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


def _pre_load(loader, model_config, *, checkpoint_id="rev1", group=_GROUP):
    return snapshot_loader.pre_load_check(
        loader=loader,
        model_config=model_config,
        checkpoint_id=checkpoint_id,
        device_config=_CPU,
        group=group,
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
        self.fetches = 0

    def publish(self, *, storages):
        self.published = {name: t.clone() for name, t in storages.items()}
        storages.clear()
        return {"kind": "memory"}

    def fetch(self, *, manifest, device):
        self.fetches += 1
        if self.fetch_error is not None:
            raise self.fetch_error
        return {name: self.published[name].clone() for name in manifest.storages}

    def close(self):
        self.closed = True


class _Source(snapshot_loader.SnapshotLoader):
    def __init__(
        self,
        load_config,
        found=None,
        owner=False,
        *,
        staged=None,
        commits=None,
        max_pending_wait=300.0,
    ):
        super().__init__(load_config, owner=owner, max_pending_wait=max_pending_wait)
        self.found = found
        self.staged = {} if staged is None else staged
        self.commits = {} if commits is None else commits

    def download_model(self, model_config):
        pass

    def _find(self, *, tag, group_tag, device_config):
        if self.found is not None:
            return self.found
        version = self.commits.get(group_tag)
        return self.staged.get((tag, version))

    def _stage(self, *, tag, manifest):
        self.staged[tag, manifest.version] = (manifest, self.publish_transport)

    def _announce(self, *, tag, version):
        self.commits[tag] = version

    def _withdraw(self, *, tag, manifest):
        self.staged.pop((tag, manifest.version), None)

    def _retract(self, *, tag, version):
        if self.commits.get(tag) == version:
            self.commits.pop(tag)


def _load_tiny_model(loader, *, model_config, device_config):
    loader_mod._initialize_model(model_config, loader.load_config)
    model = loader_mod._initialize_model(model_config, loader.load_config)
    model.post_load_weights()  # stands in for loading weights
    return model


def _spec(tp_rank=0, tp_size=2):
    """The LoadSpec that _pre_load builds on rank ``tp_rank``."""
    with _topology(tp_rank, tp_size):
        return build_load_spec(
            _model_config(), checkpoint_id="rev1", device=_CPU.device
        )


def _spec_digest(tp_rank=0, tp_size=2):
    return _spec(tp_rank, tp_size).digest()


def _manifest(*, tp_rank=0, version="v1", storages=None):
    return SnapshotManifest(
        spec_digest=_spec_digest(tp_rank),
        version=version,
        storages={} if storages is None else storages,
        load_diff=b"tables",
        attestation=b"attestation",
        source={},
    )


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
                preflight=lambda **kwargs: None,
            ),
            mock.patch.multiple(
                validator,
                record_construction=lambda m: ("check", id(m), m.post_processed),
                attest=parts["attest"],
            ),
            mock.patch.multiple(
                snapshot_loader,
                get_model_loader=mock.Mock(
                    side_effect=lambda cfg, _: loader_mod.DefaultModelLoader(cfg)
                ),
                vote=parts["vote"],
            ),
            mock.patch.object(
                loader_mod.DefaultModelLoader, "load_model", _load_tiny_model
            ),
        ):
            yield

    def _publish(self, source, transport, model_config=None, *, group=_GROUP):
        if model_config is None:
            model_config = _model_config()
        # Owner failure cases count publish votes independently of lookup and load.
        with (
            _topology(tp_rank=group.rank_in_group),
            mock.patch.object(snapshot_loader, "vote", _vote_failing_at()),
        ):
            self.assertIsNone(_pre_load(source, model_config, group=group))
            model = source.load_model(model_config=model_config, device_config=_CPU)
        exported = source.export_snapshot(model=model)
        source.publish_transport = transport
        manifest = source.publish_snapshot(
            exported=exported, transport=transport, group=group
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
            manifest.storages,
            {"proj": StorageMeta(nbytes=16, device="cpu", sharing="shared_ro")},
        )
        self.assertEqual(list(source.staged.values()), [(manifest, transport)])
        self.assertEqual(list(source.commits.values()), [manifest.version])

    def test_reader_finds_owner_snapshot_under_the_pre_load_spec(self):
        owner = _Source(LoadConfig(), owner=True)
        reader = _Source(LoadConfig(), staged=owner.staged, commits=owner.commits)
        transport = _MemoryTransport()
        model_config = _model_config()

        with (
            self._owner_parts(),
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
            self.assertEqual(len(vote.calls), 3)
            with self.assertRaises(RuntimeError):
                owner.publish_snapshot(
                    exported=exported, transport=transport, group=_GROUP
                )
        self.assertEqual(len(vote.calls), 4)
        self.assertIsNotNone(vote.calls[-1].error)
        publish.assert_not_called()
        self.assertTrue(transport.closed)
        self.assertFalse(owner.staged)
        self.assertFalse(owner.commits)

    def test_owner_rejects_a_different_publish_group(self):
        source = _Source(LoadConfig(), owner=True)
        model_config = _model_config()
        transport = _MemoryTransport()
        other_group = SimpleNamespace(rank_in_group=0)
        groups = []
        gather = _vote_failing_at()

        def vote(*, group, ballot):
            groups.append(group)
            return gather(group=group, ballot=ballot)

        with self._owner_parts(vote=vote), _topology():
            self.assertIsNone(_pre_load(source, model_config))
            model = source.load_model(model_config=model_config, device_config=_CPU)
            exported = source.export_snapshot(model=model)
            with self.assertRaisesRegex(RuntimeError, "group"):
                source.publish_snapshot(
                    exported=exported, transport=transport, group=other_group
                )
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(group is _GROUP for group in groups))
        self.assertFalse(transport.published)
        self.assertTrue(transport.closed)

    def test_every_rank_mints_the_version_from_all_attestations(self):
        gathered = [Ballot(payload=b"rank0"), Ballot(payload=b"rank1")]
        changed = [Ballot(payload=b"rank0"), Ballot(payload=b"changed")]
        versions = []
        for rank, attestation, ballots in (
            (0, b"a0", gathered),
            (1, b"a1", gathered),
            (0, b"a0", changed),
        ):
            group = SimpleNamespace(rank_in_group=rank)
            source = _Source(LoadConfig(), owner=True)
            with self._owner_parts(
                attest=lambda **kwargs: attestation,
                vote=lambda *, group, ballot: ballots,
            ):
                _, manifest = self._publish(source, _MemoryTransport(), group=group)
            self.assertEqual(manifest.spec_digest, _spec_digest(rank))
            self.assertEqual(bool(source.commits), rank == 0)
            versions.append(manifest.version)
        self.assertEqual(versions[0], versions[1])
        self.assertNotEqual(versions[0], versions[2])

    def test_owner_failure_on_any_rank_leaves_nothing_findable(self):
        def export(*, model, constructed):
            raise ValueError("unsupported tensor")

        cases = {
            "own export fails": (dict(export=export), "unsupported tensor", 1),
            "peer fails to prepare": (dict(vote=_vote_failing_at(1)), "vote 1", 1),
            "peer fails to stage": (dict(vote=_vote_failing_at(2)), "vote 2", 2),
            "commit fails": (dict(vote=_vote_failing_at(3)), "vote 3", 3),
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
                self.assertFalse(source.commits)
                self.assertFalse(source.staged)

    def test_owner_exposes_the_snapshot_only_after_the_stage_vote(self):
        source = _Source(LoadConfig(), owner=True)
        transport = _MemoryTransport()
        public = []
        staged = []

        def vote(*, group, ballot):
            public.append(bool(source.commits))
            staged.append(bool(source.staged))
            return [ballot]

        with self._owner_parts(vote=vote):
            _, manifest = self._publish(source, transport)
        self.assertEqual(staged, [False, True, True])
        self.assertEqual(public, [False, False, True])
        self.assertEqual(list(source.commits.values()), [manifest.version])

    def test_owner_does_not_export_a_presharded_load(self):
        source = _Source(
            LoadConfig(fallback_load_format=LoadFormat.PRESHARDED), owner=True
        )
        model_config = _model_config()
        vote = _vote_failing_at()
        with (
            self._owner_parts(vote=vote),
            _topology(),
            mock.patch.object(
                snapshot_loader, "get_model_loader", loader_mod.get_model_loader
            ),
            mock.patch.object(
                loader_mod.PreshardedModelLoader, "load_model", _load_tiny_model
            ),
        ):
            _pre_load(source, model_config)
            model = source.load_model(model_config=model_config, device_config=_CPU)
            self.assertTrue(model.post_processed)
            exported = source.export_snapshot(model=model)
            self.assertIn("PreshardedModelLoader", exported.error)
            self.assertFalse(exported.storages)
            with self.assertRaises(RuntimeError):
                source.publish_snapshot(
                    exported=exported, transport=_MemoryTransport(), group=_GROUP
                )
        self.assertIsNotNone(vote.calls[-1].error)
        self.assertFalse(source.staged)
        self.assertFalse(source.commits)

    def test_reader_raw_load_records_nothing(self):
        def record(model):
            raise AssertionError("only an owner records construction")

        with (
            self._owner_parts(),
            mock.patch.object(validator, "record_construction", record),
            mock.patch.object(load_diff, "record_construction", record),
            _topology(),
        ):
            source = _Source(LoadConfig())
            model_config = _model_config()
            self.assertIsNone(_pre_load(source, model_config))
            model = source.load_model(model_config=model_config, device_config=_CPU)
        self.assertFalse(model.training)

    def _read(
        self,
        *,
        preflight_error=None,
        fetch_error=None,
        install_error=None,
        validate_error=None,
        failing_vote=None,
    ):
        manifest = _manifest(storages={"proj": StorageMeta(nbytes=16, device="cpu")})
        self.transport = _MemoryTransport(fetch_error=fetch_error)
        self.transport.published = {"proj": torch.zeros(16, dtype=torch.uint8)}
        self.source = _Source(LoadConfig(), found=(manifest, self.transport))
        self.vote = _vote_failing_at(failing_vote)
        model_config = _model_config()

        def install(*, skeleton, manifest, storages, dry_run=False):
            self.assertTrue(skeleton.proj.weight.is_meta)
            if dry_run:
                self.assertEqual(self.transport.fetches, 0)
                self.assertTrue(storages["proj"].is_meta)
                self.assertEqual(
                    storages["proj"].nbytes, manifest.storages["proj"].nbytes
                )
                self.preflight_skeleton = skeleton
                if preflight_error is not None:
                    raise preflight_error
                return skeleton
            self.assertEqual(len(self.vote.calls), 1)
            self.assertFalse(storages["proj"].is_meta)
            skeleton.proj.weight = nn.Parameter(
                storages["proj"].view(torch.float32).reshape(2, 2)
            )
            self.skeleton = skeleton
            if install_error is not None:
                raise install_error
            return skeleton

        def validate(*, model, storages, attestation, constructed):
            if validate_error is not None:
                raise validate_error
            self.assertEqual(constructed, ("check", True))
            self.assertFalse(model.training)

        with (
            _topology(),
            mock.patch.object(snapshot_loader, "vote", self.vote),
            mock.patch.multiple(
                validator,
                record_construction=lambda m: ("check", m.proj.weight.is_meta),
                validate=validate,
            ),
            mock.patch.object(load_diff, "install", install),
        ):
            self.assertIs(_pre_load(self.source, model_config), manifest)
            return self.source.load_model(model_config=model_config, device_config=_CPU)

    def test_reader_installs_without_post_processing(self):
        model = self._read()
        self.assertIs(model, self.preflight_skeleton)
        self.assertFalse(model.post_processed)
        self.assertFalse(model.proj.quant_method.processed)
        self.assertEqual(len(self.vote.calls), 2)
        self.source.close()
        self.assertFalse(self.transport.closed)
        torch.testing.assert_close(model.proj.weight, torch.zeros(2, 2))

    def test_reader_preflight_failure_votes_before_fetch(self):
        for failures, error in (
            ({"preflight_error": ValueError("bad manifest")}, "bad manifest"),
            ({"failing_vote": 1}, "vote 1"),
        ):
            with self.subTest(failures=failures):
                with self.assertRaisesRegex(RuntimeError, error):
                    self._read(**failures)
                self.assertEqual(len(self.vote.calls), 1)
                self.assertEqual(self.transport.fetches, 0)
                self.assertTrue(self.transport.closed)
                with self.assertRaisesRegex(RuntimeError, "pre_load_check"):
                    self.source.load_model(
                        model_config=_model_config(), device_config=_CPU
                    )

    def test_reader_preflight_rejects_explicit_cuda_allocation(self):
        class ModelWithCudaAllocation(_TinyModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.buffer = torch.empty(1, device="cuda")

        with (
            mock.patch.object(
                loader_mod,
                "get_model_architecture",
                return_value=(ModelWithCudaAllocation, "Tiny"),
            ),
            self.assertRaisesRegex(RuntimeError, "allocate accelerator memory"),
        ):
            self._read()
        self.assertEqual(len(self.vote.calls), 1)
        self.assertEqual(self.transport.fetches, 0)
        self.assertTrue(self.transport.closed)

    def test_reader_load_failure_votes_and_keeps_bound_storage(self):
        for failures, error, mapped in (
            ({"fetch_error": RuntimeError("fetch")}, "fetch", False),
            ({"install_error": RuntimeError("install")}, "install", True),
            ({"validate_error": RuntimeError("validate")}, "validate", True),
            ({"failing_vote": 2}, "vote 2", True),
        ):
            with self.subTest(failures=failures):
                with self.assertRaisesRegex(RuntimeError, error):
                    self._read(**failures)
                self.assertEqual(len(self.vote.calls), 2)
                if failures.get("failing_vote") is None:
                    self.assertIn(error, self.vote.calls[-1].error)
                self.source.close()
                self.assertEqual(self.transport.closed, not mapped)
                if mapped:
                    torch.testing.assert_close(
                        self.skeleton.proj.weight, torch.zeros(2, 2)
                    )

    def test_reader_rejects_another_ranks_manifest_before_fetch(self):
        transport = _MemoryTransport()
        source = _Source(LoadConfig(), found=(_manifest(tp_rank=1), transport))
        vote = _vote_failing_at()
        with (
            _topology(tp_rank=0),
            mock.patch.object(snapshot_loader, "vote", vote),
            self.assertRaisesRegex(RuntimeError, "published for LoadSpec"),
        ):
            _pre_load(source, _model_config())
        self.assertIn("published for LoadSpec", vote.calls[0].error)
        self.assertEqual(transport.fetches, 0)
        self.assertTrue(transport.closed)

    def test_pre_load_check_votes_on_every_outcome_but_the_rank(self):
        ballots = []
        manifest = _manifest(tp_rank=1)
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
        with (
            mock.patch.object(
                snapshot_loader, "vote", lambda *, group, ballot: ballots.append(ballot)
            ),
            mock.patch.object(load_diff, "preflight"),
            mock.patch.object(validator, "record_construction"),
        ):
            for loader, rank, size in cases:
                with _topology(rank, size):
                    _pre_load(loader, _model_config())
        for i in (0, 1, 3, 4):
            self.assertIsNone(ballots[i].error)
        self.assertEqual(ballots[0].agree, ballots[1].agree)
        self.assertIn("daemon socket refused", ballots[2].error)
        # Rank 1 accepted its own manifest: the same load as rank 0, at version v1.
        group, version = msgspec.msgpack.decode(ballots[3].agree)
        self.assertEqual(group, msgspec.msgpack.decode(ballots[0].agree)[0])
        self.assertEqual(version, "v1")
        self.assertNotEqual(ballots[4].agree, ballots[0].agree)

        # When the ranks reject the load, a rank that found a snapshot releases it.
        transport = _MemoryTransport()
        rejected = _Source(LoadConfig(), found=(manifest, transport))
        with (
            _topology(tp_rank=1),
            mock.patch.object(load_diff, "preflight"),
            mock.patch.object(validator, "record_construction"),
            mock.patch.object(snapshot_loader, "vote", vote := _vote_failing_at(1)),
        ):
            with self.assertRaises(RuntimeError):
                _pre_load(rejected, _model_config())
        self.assertIsNone(vote.calls[0].error)  # rejected by a peer, not locally
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

    def test_only_parameter_storages_are_shared(self):
        def export(*, model, constructed):
            weight = model.proj.weight.data.view(torch.uint8)
            return {
                "proj": weight,
                "scale": torch.zeros(4, dtype=torch.uint8),
                "gpu": torch.empty(4, dtype=torch.uint8, device="meta"),
            }, b""

        with (
            self._owner_parts(export=export),
            mock.patch.object(
                snapshot_loader.current_platform,
                "get_device_uuid",
                return_value="GPU-0",
            ),
        ):
            _, manifest = self._publish(
                _Source(LoadConfig(), owner=True), _MemoryTransport()
            )
        storages = manifest.storages
        self.assertEqual(storages["proj"].sharing, "shared_ro")
        self.assertEqual(storages["scale"].sharing, "private")
        self.assertIsNone(storages["scale"].device_uuid)  # CPU memory has no GPU
        self.assertEqual(storages["gpu"].device_uuid, "GPU-0")

    def test_snapshot_support_reads_the_spec(self):
        spec = _spec()
        self.assertIsNone(snapshot_loader.snapshot_support(spec))
        for unverified in (
            msgspec.structs.replace(spec, quant_method="fp8"),
            msgspec.structs.replace(spec, enable_eplb=True),
        ):
            self.assertIsNotNone(snapshot_loader.snapshot_support(unverified))

    def test_an_unverified_load_is_raw_on_every_rank_and_never_publishes(self):
        source, transport = _Source(LoadConfig(), owner=True), _MemoryTransport()
        with (
            self._owner_parts(),
            _topology(),
            mock.patch.object(snapshot_loader, "snapshot_support", return_value="awq"),
            mock.patch.object(source, "_find") as find,
        ):
            self.assertIsNone(_pre_load(source, _model_config()))
            model = source.load_model(model_config=_model_config(), device_config=_CPU)
            exported = source.export_snapshot(model=model)
            with self.assertRaisesRegex(RuntimeError, "awq"):
                source.publish_snapshot(
                    exported=exported, transport=transport, group=_GROUP
                )
        find.assert_not_called()
        self.assertFalse(transport.published)
        self.assertFalse(source.staged)

    def test_lookup_waits_for_a_pending_owner(self):
        manifest, transport = _manifest(), _MemoryTransport()
        source = _Source(LoadConfig(), max_pending_wait=10)
        source._find = mock.Mock(
            side_effect=[snapshot_loader.Pending(1.0), (manifest, transport)]
        )
        with mock.patch.object(snapshot_loader.time, "sleep") as sleep:
            self.assertIs(source.lookup(spec=_spec(), device_config=_CPU), manifest)
        sleep.assert_called_once_with(1.0)

    def test_lookup_waits_on_pending_until_the_deadline(self):
        now = [0.0]
        source = _Source(
            LoadConfig(), found=snapshot_loader.Pending(5.0), max_pending_wait=12
        )
        with (
            mock.patch.object(snapshot_loader.time, "monotonic", lambda: now[0]),
            mock.patch.object(
                snapshot_loader.time,
                "sleep",
                side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds),
            ) as sleep,
            self.assertRaisesRegex(RuntimeError, "did not publish within 12 s"),
        ):
            source.lookup(spec=_spec(), device_config=_CPU)
        # It uses the time left, and never sleeps past the deadline.
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5.0, 5.0, 2.0])

    def test_a_source_that_cannot_serve_stops_every_rank(self):
        source = _Source(LoadConfig())
        source._find = mock.Mock(side_effect=RuntimeError("daemon out of memory"))
        vote = _vote_failing_at()
        with (
            _topology(),
            mock.patch.object(snapshot_loader, "vote", vote),
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            _pre_load(source, _model_config())
        self.assertIn("out of memory", vote.calls[0].error)
        # No rank falls back to a raw load.
        with self.assertRaisesRegex(RuntimeError, "pre_load_check"):
            source.load_model(model_config=_model_config(), device_config=_CPU)

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
