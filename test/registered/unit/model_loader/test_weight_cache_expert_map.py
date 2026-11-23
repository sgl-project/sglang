"""The expert map of a load-diff snapshot (weight_cache/expert_map.py): what a
reader refuses before anything is attached, and that a reader that took the
owner's map sends every token to a slot that holds the token's expert.

Each process is emulated in this interpreter: a published dummy config with
the settings the argument pipeline resolves for a real model (for 'dummy' it
stops before it resolves --ep-dispatch-algorithm), an EP rank, and the map
the runner, or the daemon, sets before it loads. The model registry lookup of
an expert layout's shape is patched: no model is looked up.
"""

import contextlib
import functools
import json
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.model_executor.model_runner_components import (
    load_model_utils,
    moe_ep_setup,
)
from sglang.srt.runtime_context import get_context, get_parallel, get_resources
from sglang.srt.weight_cache import expert_map, load_diff
from sglang.srt.weight_cache.load_diff_format import (
    ExpertMap,
    SnapshotError,
    decode_manifest,
    expert_map_digest,
)
from sglang.srt.weight_cache.schema import SnapshotManifest, StorageMeta
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_NUM_LAYERS, _NUM_EXPERTS = 2, 4

# Initial layouts (--init-expert-location) of two layers of four experts: the
# logical expert in each physical slot.
_TRIVIAL = [[0, 1, 2, 3], [0, 1, 2, 3]]
_A = [[3, 2, 1, 0], [1, 0, 3, 2]]
_B = [[0, 1, 3, 2], [2, 3, 0, 1]]

# ... and with two more slots (--ep-num-redundant-experts 2): two experts of
# each layer have a replica.
_REDUNDANT = dict(ep_num_redundant_experts=2)
_REPLICAS_A = [[0, 1, 2, 3, 3, 0], [3, 3, 2, 1, 0, 0]]
_REPLICAS_B = [[1, 1, 0, 2, 3, 2], [0, 1, 2, 3, 0, 1]]


@contextlib.contextmanager
def _process(layout=None, *, moe=True, dispatch=None, ep_size=1, **settings):
    """An emulated process up to its expert map: its config, EP rank 0, and
    the map set before it loads (``layout``: its --init-expert-location;
    trivial without). ``dispatch``: the --ep-dispatch-algorithm it resolved
    ("static" and "lp" take an A2A backend); ``settings``: other resolved
    fields; ``moe=False``: a model without experts, which gets no map."""
    if layout is not None:
        settings["init_expert_location"] = json.dumps(
            {"physical_to_logical_map": layout}
        )
    with (
        get_context().override_server_args(
            device="cpu",
            tp_size=ep_size,
            ep_size=ep_size,
            moe_a2a_backend="deepep" if dispatch in ("static", "lp") else "none",
            ep_dispatch_algorithm=dispatch,
            **settings,
        ),
        get_parallel().override(moe_ep_rank=0),
        get_resources().override(expert_location_metadata=None),
    ):
        if moe:
            # As ModelRunner.maybe_init_expert_location_metadata, and the
            # daemon's _initialize_expert_location_metadata.
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    model_config=None, moe_ep_rank=0
                )
            )
        yield


def _exported(layout=None, **process):
    """What an owner that placed its experts by ``layout`` puts in its manifest."""
    with _process(layout, **process):
        return expert_map.export_expert_map()


def _process_map():
    return get_global_expert_location_metadata().physical_to_logical_map_cpu.tolist()


class _Case(CustomTestCase):
    def setUp(self):
        # Stands in for the model registry lookup of an expert layout's shape.
        patcher = mock.patch.object(
            ModelConfigForExpertLocation,
            "from_model_config",
            return_value=ModelConfigForExpertLocation(
                num_layers=_NUM_LAYERS, num_logical_experts=_NUM_EXPERTS
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _runs_the_experts_asked_for(self, held, *, infos=None):
        """Every token runs in a slot that holds the expert the router picked
        for it. ``held``: the expert in each slot (where the weights are), by
        layer; ``infos``: dispatch infos made earlier, as a layer keeps them.
        Returns the slots used, by layer."""
        held = torch.as_tensor(held).to(torch.int64)
        used = []
        for layer_id in range(_NUM_LAYERS):
            info = (
                ExpertLocationDispatchInfo.init_new(layer_id)
                if infos is None
                else infos[layer_id]
            )
            # Every logical expert in each of three rows: a replica is picked
            # by row.
            logical = torch.arange(_NUM_EXPERTS).repeat(3, 1)
            physical = topk_ids_logical_to_physical(logical, info)
            self.assertTrue(
                torch.equal(held[layer_id][physical], logical),
                (layer_id, physical.tolist()),
            )
            used.append(set(physical.flatten().tolist()))
        return used


class TestBeforeAnythingIsAttached(_Case):
    """check_expert_map: nothing to take, a map to take, or a refusal."""

    def test_nothing_to_take(self):
        with _process(moe=False):
            self.assertIsNone(expert_map.check_expert_map(None))
        for layout in (None, _A):
            owner = _exported(layout)
            # The same values are nothing to take for any engine: one that
            # does not dispatch by its map, one with CPU experts, a joiner.
            for engine in (
                dict(),
                dict(dispatch="dynamic"),
                dict(dispatch="dynamic", kt_weight_path="/kt"),
                dict(dispatch="dynamic", ep_join_mode="scale"),
            ):
                with self.subTest(layout=layout, **engine), _process(layout, **engine):
                    self.assertIsNone(expert_map.check_expert_map(owner))

    def test_another_layout_of_the_same_shape_is_taken(self):
        owner, trivial = _exported(_A), _exported()
        for dispatch in ("dynamic", "static", "lp"):
            with self.subTest(dispatch=dispatch), _process(_B, dispatch=dispatch):
                theirs = expert_map.check_expert_map(owner)
                self.assertEqual(theirs.dtype, torch.int64)
                self.assertEqual(theirs.tolist(), _A)
                # The check changes nothing: the engine builds by its own map.
                self.assertEqual(_process_map(), _B)
        # An owner started without a layout serves an engine with one.
        with _process(_B, dispatch="dynamic"):
            theirs = expert_map.check_expert_map(trivial)
            self.assertEqual(theirs.tolist(), _TRIVIAL)

    def test_a_map_on_one_side_only(self):
        owner = _exported(_A)
        with _process(_B, dispatch="dynamic"):
            with self.assertRaisesRegex(SnapshotError, "the owner's snapshot has none"):
                expert_map.check_expert_map(None)
        with _process(moe=False):
            with self.assertRaisesRegex(SnapshotError, "this process has none"):
                expert_map.check_expert_map(owner)

    def test_another_shape(self):
        more_layers = _A + [[0, 1, 2, 3]]
        owners = {
            "more slots": _exported(_REPLICAS_A, **_REDUNDANT),
            "more layers": ExpertMap(
                num_logical_experts=_NUM_EXPERTS,
                ep_size=1,
                rows=more_layers,
                digest=expert_map_digest(more_layers),
            ),
            # The same rows over two EP ranks: other slots belong to each rank.
            "another EP size": _exported(_A, ep_size=2),
            "fewer logical experts": ExpertMap(
                num_logical_experts=2,
                ep_size=1,
                rows=[[0, 1, 1, 0]] * 2,
                digest=expert_map_digest([[0, 1, 1, 0]] * 2),
            ),
        }
        for name, owner in owners.items():
            with self.subTest(name), _process(_B, dispatch="dynamic"):
                with self.assertRaisesRegex(SnapshotError, "another shape"):
                    expert_map.check_expert_map(owner)

    def test_an_engine_whose_layers_do_not_remap(self):
        """Such an engine runs each token in the slot of its expert's number,
        whatever map the process holds: it cannot follow another layout."""
        owner = _exported(_A)
        refusal = r"8 of 8 physical slots.*do not pick a token's slot through the map"
        # No dispatch algorithm: the layers get no dispatch info.
        with _process(_B):
            self.assertIsNone(ExpertLocationDispatchInfo.init_new(0))
            with self.assertRaisesRegex(SnapshotError, refusal):
                expert_map.check_expert_map(owner)
        # An algorithm, but a trivial layout, no EPLB and no redundant
        # experts: the layers skip the remap (is_expert_remapping_enabled).
        with _process(dispatch="dynamic"):
            with self.assertRaisesRegex(SnapshotError, refusal):
                expert_map.check_expert_map(owner)
        # ... while each of the three makes them remap.
        for name, engine in (
            ("a layout", dict(layout=_B)),
            ("EPLB", dict(enable_eplb=True)),
        ):
            with self.subTest(name), _process(dispatch="dynamic", **engine):
                self.assertIsNotNone(expert_map.check_expert_map(owner))
        owner = _exported(_REPLICAS_A, **_REDUNDANT)
        with _process(dispatch="dynamic", **_REDUNDANT):
            self.assertIsNotNone(expert_map.check_expert_map(owner))

    def test_an_engine_that_cannot_follow_another_layout(self):
        owner = _exported(_A)
        engines = {
            # KTransformers CPU experts are placed by the engine's own map.
            "--kt-weight-path": dict(kt_weight_path="/kt"),
            # Its layout is computed for a shifted rank and then replaced by
            # the one of the ranks it joins.
            "scale joiner": dict(ep_join_mode="scale"),
        }
        for refusal, engine in engines.items():
            with self.subTest(refusal), _process(_B, dispatch="dynamic", **engine):
                with self.assertRaisesRegex(SnapshotError, refusal):
                    expert_map.check_expert_map(owner)
                self.assertEqual(_process_map(), _B)


class TestOnceTheTensorsAreInstalled(_Case):
    """adopt_expert_map: the process dispatches by the owner's map."""

    def _adopt(self, owner):
        theirs = expert_map.check_expert_map(owner)
        with self.assertLogs(expert_map.logger, "WARNING") as logs:
            report = expert_map.adopt_expert_map(theirs)
        return report, logs.output

    def test_the_engine_dispatches_by_the_owners_map(self):
        owner = _exported(_A)
        for dispatch in ("dynamic", "static"):
            with self.subTest(dispatch=dispatch), _process(_B, dispatch=dispatch):
                metadata = get_global_expert_location_metadata()
                # By its own map the engine would run other experts than the
                # router picked, on the owner's weights.
                self._runs_the_experts_asked_for(_B)
                with self.assertRaises(AssertionError):
                    self._runs_the_experts_asked_for(_A)
                # What a layer that kept its dispatch info holds.
                kept = [
                    ExpertLocationDispatchInfo.init_new(layer_id)
                    for layer_id in range(_NUM_LAYERS)
                ]

                report, logged = self._adopt(owner)

                # The same object (the recorder keeps it), with the owner's map.
                self.assertIs(get_global_expert_location_metadata(), metadata)
                self.assertEqual(_process_map(), _A)
                self.assertEqual(expert_map.export_expert_map(), owner)
                self._runs_the_experts_asked_for(_A)
                self._runs_the_experts_asked_for(_A, infos=kept)
                self.assertEqual(
                    report,
                    {
                        "digest": owner.digest.hex(),
                        "changed_physical_slots": 8,
                        "physical_slots": 8,
                        "lplb_solvers_rebuilt": False,
                    },
                )
                (line,) = logged
                self.assertIn("adopted the owner's expert map", line)
                # Nothing left to take.
                self.assertIsNone(expert_map.check_expert_map(owner))

    def test_the_replicas_are_the_owners(self):
        """With redundant experts the engine spreads an expert's tokens over
        the slots the owner put that expert's replicas in."""
        owner = _exported(_REPLICAS_A, **_REDUNDANT)
        with _process(_REPLICAS_B, dispatch="dynamic", **_REDUNDANT):
            metadata = get_global_expert_location_metadata()
            self._adopt(owner)
            # Experts 0 and 3 have two slots in each layer of the owner's
            # map; experts 1 and 2 had them in the engine's.
            self.assertEqual(
                metadata.logical_to_all_physical_map_num_valid.tolist(),
                [[2, 1, 1, 2], [2, 1, 1, 2]],
            )
            used = self._runs_the_experts_asked_for(_REPLICAS_A)
            # Every replica gets tokens.
            self.assertEqual(used, [set(range(6)), set(range(6))])

    def test_the_lplb_solvers_are_built_again(self):
        """The runner built them from the engine's own map before the load."""
        owner = _exported(_A)
        for dispatch, built in (("lp", 1), ("dynamic", 0)):
            with self.subTest(dispatch=dispatch), _process(_B, dispatch=dispatch):
                seen = []
                # The solvers need the CUDA LP kernels: record the map they
                # would be built from.
                with mock.patch.object(
                    moe_ep_setup,
                    "rebuild_lplb_solvers",
                    lambda: seen.append(_process_map()),
                ):
                    report, _ = self._adopt(owner)
                self.assertEqual(seen, [_A] * built)
                self.assertEqual(report["lplb_solvers_rebuilt"], bool(built))

    def test_a_map_that_did_not_take_ends_the_load(self):
        owner = _exported(_A)
        with _process(_B, dispatch="dynamic"):
            theirs = expert_map.check_expert_map(owner)
            with mock.patch.object(ExpertLocationMetadata, "adopt_layout"):
                with self.assertRaisesRegex(SnapshotError, "not the owner's"):
                    expert_map.adopt_expert_map(theirs)


class Experts(nn.Module):
    """Stands in for a model's MoE layers: row l holds, for each physical
    slot of layer l, the id of the expert whose weights are there."""

    def __init__(self):
        super().__init__()
        self.rows = nn.Parameter(
            torch.full((_NUM_LAYERS, _NUM_EXPERTS), -1.0), requires_grad=False
        )


def _load(model):
    """Place each checkpoint expert as FusedMoE.weight_loader does: in the
    slots the process's map gives it."""
    metadata = get_global_expert_location_metadata()
    with torch.no_grad():
        for layer_id in range(_NUM_LAYERS):
            for expert_id in range(_NUM_EXPERTS):
                slots = metadata.logical_to_all_physical(layer_id, expert_id)
                model.rows[layer_id, slots] = float(expert_id)


def _snapshot(layout=None, **process):
    """What an owner that loaded by ``layout`` published: the manifest and
    its storages."""
    with _process(layout, **process):
        owner = Experts()
        constructed = load_diff.record_construction(owner)
        if process.get("moe", True):
            _load(owner)
        storages, payload = load_diff.export(model=owner, constructed=constructed)
    manifest = SnapshotManifest(
        version="v",
        storages={
            name: StorageMeta(nbytes=t.nbytes, device=t.device.type)
            for name, t in storages.items()
        },
        load_diff=payload,
        attestation=b"",
        source={},
    )
    return manifest, storages


def _skeleton():
    with torch.device("meta"):
        return Experts()


def _install(skeleton, manifest, storages):
    """Through the interface, on copies of ``storages``."""
    return load_diff.install(
        skeleton=skeleton,
        manifest=manifest,
        storages={name: t.clone() for name, t in storages.items()},
    )


class TestThroughLoadDiff(_Case):
    """load_diff.export records the owner's map; load_diff.install checks it
    before it attaches anything and takes it once the result is checked."""

    def test_the_manifest_carries_the_map_the_owner_loaded_by(self):
        manifest, _ = _snapshot(_A)
        carried = decode_manifest(manifest.load_diff).expert_map
        self.assertEqual(carried.rows, _A)
        self.assertEqual((carried.num_logical_experts, carried.ep_size), (4, 1))
        # What the ranks of one load agree on before any byte moves.
        self.assertEqual(
            load_diff.owner_expert_map_digest(manifest), expert_map_digest(_A)
        )
        other, _ = _snapshot(_B)
        self.assertNotEqual(
            load_diff.owner_expert_map_digest(other),
            load_diff.owner_expert_map_digest(manifest),
        )
        # A model without experts carries none.
        dense, _ = _snapshot(moe=False)
        self.assertIsNone(decode_manifest(dense.load_diff).expert_map)
        self.assertEqual(load_diff.owner_expert_map_digest(dense), b"")

    def test_a_reader_with_another_layout_dispatches_by_the_owners(self):
        manifest, storages = _snapshot(_A)
        with _process(_B, dispatch="dynamic"):
            with self.assertLogs(expert_map.logger, "WARNING"):
                model = _install(_skeleton(), manifest, storages)
            self.assertEqual(_process_map(), _A)
            # On the weights it installed, not on a layout stated apart.
            self._runs_the_experts_asked_for(model.rows.data)
        # The same layout loads as before, with nothing to take.
        with _process(_A, dispatch="dynamic"):
            with self.assertNoLogs(expert_map.logger, "WARNING"):
                model = _install(_skeleton(), manifest, storages)
            self._runs_the_experts_asked_for(model.rows.data)

    def test_a_refusal_comes_before_anything_is_attached(self):
        manifest, storages = _snapshot(_A)
        # An engine without a dispatch algorithm cannot follow another layout.
        with _process(_B):
            skeleton = _skeleton()
            with self.assertRaisesRegex(SnapshotError, "cannot take the owner's"):
                _install(skeleton, manifest, storages)
            self.assertTrue(skeleton.rows.is_meta)
            self.assertEqual(_process_map(), _B)

    def test_the_map_is_taken_only_once_the_install_is_checked(self):
        manifest, storages = _snapshot(_A)
        (name,) = storages
        flipped = {name: storages[name].clone()}
        flipped[name][0] ^= 1
        with _process(_B, dispatch="dynamic"):
            with self.assertRaisesRegex(SnapshotError, "not the one the owner"):
                _install(_skeleton(), manifest, flipped)
            self.assertEqual(_process_map(), _B)


class _Group:
    """Stands in for a process group of ``len(others) + 1`` ranks, seen from
    ``rank``: all_gather_object returns every rank's object in rank order."""

    cpu_group = None

    def __init__(self, rank=0, others=()):
        self.rank, self.others, self.gathered = rank, list(others), []

    def all_gather_object(self, obj):
        self.gathered = self.others[: self.rank] + [obj] + self.others[self.rank :]
        return self.gathered


@contextlib.contextmanager
def _after_a_cache_hit(layout, manifest, storages):
    """A rank that set ``layout`` and then installed the owner's snapshot."""
    with _process(layout, dispatch="dynamic"):
        _install(_skeleton(), manifest, storages)
        yield


class TestTheRanksAgree(_Case):
    """agree_on_expert_map: once every rank is loaded, they all dispatch by
    the same map, or every rank raises the same report."""

    def _agree(self, *ranks):
        """Run the agreement on every rank of one engine. Each of ``ranks``
        makes the context of a rank as it is after its load. Returns each
        rank's error message, or None."""
        mine = []
        for rank in ranks:
            with rank():
                alone = _Group()
                expert_map.agree_on_expert_map(alone)
                mine += alone.gathered
        errors = []
        for i, rank in enumerate(ranks):
            others = mine[:i] + mine[i + 1 :]
            with rank():
                try:
                    expert_map.agree_on_expert_map(_Group(i, others))
                    errors.append(None)
                except SnapshotError as e:
                    errors.append(str(e))
        return errors

    def test_the_same_map_or_none_on_every_rank_passes(self):
        with_a = functools.partial(_process, _A, dispatch="dynamic")
        self.assertEqual(self._agree(with_a, with_a), [None, None])
        # A model without experts: no rank has a map.
        dense = functools.partial(_process, moe=False)
        self.assertEqual(self._agree(dense, dense), [None, None])

    def test_a_rank_that_fell_back_to_disk_fails_every_rank(self):
        """In client mode a rank whose daemon is absent loads from disk and
        keeps its own map, while the ranks that hit the cache took the
        daemon's: an expert's tokens would run in two ranks' slots, or none."""
        manifest, storages = _snapshot(_A)
        hit = functools.partial(_after_a_cache_hit, _B, manifest, storages)
        fell_back = functools.partial(_process, _B, dispatch="dynamic")
        on_hit, on_fallback = self._agree(hit, fell_back)
        self.assertEqual(on_hit, on_fallback)
        self.assertIn(f"ranks [0]: {expert_map_digest(_A).hex()}", on_hit)
        self.assertIn(f"ranks [1]: {expert_map_digest(_B).hex()}", on_hit)
        self.assertIn("--init-expert-location", on_hit)
        # Every rank on the cache, and a layout the engine also set: fine.
        self.assertEqual(self._agree(hit, hit), [None, None])

    def test_a_map_on_some_ranks_only_fails_every_rank(self):
        with_a = functools.partial(_process, _A, dispatch="dynamic")
        dense = functools.partial(_process, moe=False)
        errors = self._agree(with_a, dense, with_a)
        self.assertEqual(len(set(errors)), 1)
        self.assertIn("ranks [1]: no expert map", errors[0])
        self.assertIn(f"ranks [0, 2]: {expert_map_digest(_A).hex()}", errors[0])

    def test_it_follows_the_post_load_barrier_only_with_the_weight_cache(self):
        for mode in ("off", "client", "daemon"):
            group = _Group()
            with (
                self.subTest(mode=mode),
                _process(_A, dispatch="dynamic", weight_cache_mode=mode),
                get_parallel().override(tp_group=group),
                # The barrier needs a real process group.
                mock.patch.object(load_model_utils.dist, "monitored_barrier"),
            ):
                load_model_utils.dist_barrier_after_load(
                    elastic_ep_backend=None, tp_rank=0
                )
            self.assertEqual(bool(group.gathered), mode != "off")


if __name__ == "__main__":
    unittest.main()
