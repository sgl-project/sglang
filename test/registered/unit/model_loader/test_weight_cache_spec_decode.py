"""Speculative decoding with the weight cache: the daemon loads the draft
model after the target, as the engine's draft worker builds it, and serves it
under the draft role (weight_cache/draft.py and the draft paths of daemon.py
and ipc_loader.py).

Covered on CPU: which setups are refused, the scope the daemon builds the
draft in (checked against the workers' source), one copy of what the draft
takes from the target (toy models and the real draft classes, through the
worker's own init_lm_head), and the daemon serving both models to an engine
over a socket pair: load-diff manifests, a storage the two share exported
once, and a RoPE cache that the target's runner grows and the draft shares
(real get_rope)."""

import contextlib
import inspect
import re
import socket
import threading
import types
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.weight_cache import daemon as daemon_module
from sglang.srt.weight_cache import draft as wc_draft
from sglang.srt.weight_cache import ipc_loader, load_diff
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.protocol import (
    DRAFT_ROLE,
    PROTOCOL_VERSION,
    TARGET_ROLE,
    CacheConfig,
    handshake_with_daemon,
    recv_msg,
    send_msg,
)
from sglang.srt.weight_cache.transport import TORCH_IPC_BACKEND
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@contextlib.contextmanager
def _published(**kwargs):
    """A published runtime context (CPU) for code that reads the config bags."""
    reset_context()
    server = ServerArgs(model_path="dummy", device="cpu", tp_size=1, **kwargs)
    publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    try:
        yield server
    finally:
        reset_context()


@contextlib.contextmanager
def _no_device_sync():
    """Sharing methods empty the device cache and synchronize: CPU no-ops."""
    with (
        mock.patch("torch.cuda.empty_cache"),
        mock.patch("torch.cuda.synchronize"),
        mock.patch("torch.cuda.memory_reserved", return_value=0),
        mock.patch.object(current_platform, "empty_cache"),
        mock.patch.object(current_platform, "synchronize"),
    ):
        yield


def _ptr(t):
    return t.untyped_storage().data_ptr()


# ----------------------------------------------------------------------------
# Which setups the weight cache serves
# ----------------------------------------------------------------------------


class TestServerArgs(CustomTestCase):
    def _check(self, **kwargs):
        from sglang.srt.arg_groups.model_path_hook import handle_load_format

        handle_load_format(ServerArgs(model_path="dummy", **kwargs))

    def test_served_algorithms_are_accepted(self):
        for algo in ("EAGLE", "NEXTN", "EAGLE3", "STANDALONE", "NGRAM"):
            for mode in ("daemon", "client"):
                with self.subTest(algo=algo, mode=mode):
                    self._check(weight_cache_mode=mode, speculative_algorithm=algo)

    def test_refusals_name_their_reason(self):
        for algo in ("DFLASH", "DSPARK", "UNO", "FROZEN_KV_MTP"):
            with self.subTest(algo=algo):
                with self.assertRaisesRegex(
                    ValueError, f"speculative-algorithm {algo}"
                ):
                    self._check(weight_cache_mode="daemon", speculative_algorithm=algo)
                # Without the weight cache nothing changes
                self._check(weight_cache_mode="off", speculative_algorithm=algo)
        for extra, reason in (
            ({"enable_multi_layer_eagle": True}, "multi-layer-eagle"),
            ({"pp_size": 2}, "pp-size"),
            ({"nnodes": 2}, "nnodes"),
        ):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, reason):
                    self._check(
                        weight_cache_mode="client",
                        speculative_algorithm="EAGLE",
                        **extra,
                    )
                self._check(
                    weight_cache_mode="off", speculative_algorithm="EAGLE", **extra
                )
        # A STANDALONE draft is one model whatever the EAGLE flag says
        self._check(
            weight_cache_mode="client",
            speculative_algorithm="STANDALONE",
            enable_multi_layer_eagle=True,
        )


# ----------------------------------------------------------------------------
# Which draft the daemon loads, in which scope
# ----------------------------------------------------------------------------


class TestDraftPlan(CustomTestCase):
    def _plan(self, algo, **kwargs):
        return wc_draft.plan_draft(
            ServerArgs(model_path="dummy", speculative_algorithm=algo, **kwargs)
        )

    def test_plan_follows_create_worker(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertIsNone(self._plan(None))
        self.assertIsNone(self._plan("NGRAM"))
        for algo, standalone in (
            ("EAGLE", False),
            ("NEXTN", False),
            ("EAGLE3", False),
            ("STANDALONE", True),
        ):
            with self.subTest(algo=algo):
                plan = self._plan(algo)
                self.assertEqual(plan.standalone, standalone)
                self.assertEqual(plan.algorithm, SpeculativeAlgorithm.from_string(algo))
        for algo in ("DFLASH", "UNO"):
            with self.subTest(algo=algo), self.assertRaises(ValueError):
                self._plan(algo)
        with self.assertRaisesRegex(ValueError, "multi-layer"):
            self._plan("EAGLE", enable_multi_layer_eagle=True)

    # The worker each plan stands for; the daemon mirrors how its __init__
    # builds the draft (DraftPlan.scope)
    WORKERS = {
        False: ("sglang.srt.speculative.eagle_worker_v2", "EagleDraftWorker"),
        True: ("sglang.srt.speculative.standalone_worker_v2", "StandaloneDraftWorker"),
    }
    CONTEXTS = (
        "draft_tp_context",
        "draft_pp_context",
        "speculative_moe_backend_context",
        "speculative_moe_a2a_backend_context",
        "draft_model_build_scope",
    )

    def _worker_scope(self, standalone):
        """The contexts the worker's __init__ enters around building its draft
        (the with-block that makes the TpModelWorker), in order, from its
        source."""
        import importlib

        module, cls = self.WORKERS[standalone]
        src = inspect.getsource(getattr(importlib.import_module(module), cls).__init__)
        build = re.search(
            r"\n\s*with \((.*?)\):\n\s*self\.draft_worker = TpModelWorker\(", src, re.S
        )
        self.assertIsNotNone(build, f"{cls}.__init__ builds its draft in no with-block")
        return re.findall(r"\b(" + "|".join(self.CONTEXTS) + r")\(", build.group(1))

    def test_scope_mirrors_each_worker(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        entered = []

        def recorder(name):
            @contextlib.contextmanager
            def context(*args, **kwargs):
                entered.append(name)
                yield

            return context

        patches = [
            mock.patch(f"sglang.srt.layers.moe.utils.{n}", recorder(n))
            for n in self.CONTEXTS[2:]
        ] + [
            mock.patch(f"sglang.srt.speculative.spec_utils.{n}", recorder(n))
            for n in self.CONTEXTS[:2]
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            for standalone, algo in ((False, "EAGLE"), (True, "STANDALONE")):
                with self.subTest(algo=algo):
                    entered.clear()
                    plan = wc_draft.DraftPlan(SpeculativeAlgorithm.from_string(algo))
                    with plan.scope(layout=None):
                        pass
                    want = self._worker_scope(standalone)
                    self.assertEqual(entered, want)
                    self.assertEqual(len(want), 3 if standalone else 5)

    def test_layout_as_the_worker_resolves_it(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        with (
            mock.patch(
                "sglang.srt.speculative.draft_parallel.resolve_draft_parallel_layout",
                side_effect=lambda spec, **kw: ("layout", spec),
            ),
            mock.patch(
                "sglang.srt.speculative.draft_parallel.model_draft_parallel_spec",
                side_effect=lambda cfg, spec_algorithm: ("spec", cfg, spec_algorithm),
            ),
            mock.patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state.resolve_spec_aux_hidden_state_config",
                return_value=types.SimpleNamespace(draft_model_config="draft-config"),
            ) as aux,
        ):
            eagle = SpeculativeAlgorithm.from_string("EAGLE")
            got = wc_draft.DraftPlan(eagle).layout("server", "target-config")
            self.assertEqual(got, ("layout", ("spec", "draft-config", eagle)))
            aux.assert_called_once_with(
                server_args="server",
                model_config="target-config",
                spec_algorithm=eagle,
                is_draft_worker=False,
            )
            standalone = wc_draft.DraftPlan(
                SpeculativeAlgorithm.from_string("STANDALONE")
            )
            _, spec = standalone.layout("server", "target-config")
            self.assertEqual(spec.tp_execution.name, "TARGET_LAYOUT")


# ----------------------------------------------------------------------------
# One copy of what the draft takes from the target
# ----------------------------------------------------------------------------


class _Target(nn.Module):
    def __init__(self, vocab=8, hidden=4):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight


def _toy_draft(vocab=8, hidden=4, tied=False):
    d = _Target(vocab, hidden)
    d.layer = nn.Linear(hidden, hidden)
    for p in d.parameters():
        p.weight_loader = d.layer  # a Python attribute the draft keeps
    if tied:
        d.lm_head.weight = d.model.embed_tokens.weight
    return d


class TestShareLikeEngine(CustomTestCase):
    def _state(self, model):
        return [
            (name, id(m), dict(m._parameters), dict(m._buffers), dict(m._modules), sorted(vars(m)))
            for name, m in model.named_modules()
        ]  # fmt: skip

    def test_replaced_tensors_keep_their_objects_and_take_the_targets_data(self):
        target, draft = _Target(), _toy_draft()
        before = self._state(draft)
        embed, head = draft.model.embed_tokens.weight, draft.lm_head.weight

        def step():  # set_embed_and_head, then a flag
            del draft.model.embed_tokens.weight
            del draft.lm_head.weight
            draft.model.embed_tokens.weight = target.model.embed_tokens.weight
            draft.lm_head.weight = target.lm_head.weight
            draft.shared = True

        shared = wc_draft.share_like_engine(target, draft, step)
        self.assertEqual(shared, ["lm_head.weight", "model.embed_tokens.weight"])
        self.assertEqual(self._state(draft), before)  # objects, slots, attributes
        self.assertIs(draft.model.embed_tokens.weight, embed)
        self.assertEqual(_ptr(embed), _ptr(target.model.embed_tokens.weight))
        self.assertEqual(_ptr(head), _ptr(target.lm_head.weight))
        self.assertIs(embed.weight_loader, draft.layer)
        self.assertNotEqual(_ptr(draft.layer.weight), _ptr(target.lm_head.weight))

    def test_a_module_taken_whole(self):
        target, draft = _Target(), _toy_draft()
        head_module, head = draft.lm_head, draft.lm_head.weight

        def step():  # set_lm_head_from_target
            draft.lm_head = target.lm_head

        self.assertEqual(
            wc_draft.share_like_engine(target, draft, step), ["lm_head.weight"]
        )
        self.assertIs(draft.lm_head, head_module)
        self.assertIs(draft.lm_head.weight, head)
        self.assertEqual(_ptr(head), _ptr(target.lm_head.weight))

    def test_what_the_draft_still_holds_stays_its_own(self):
        # A head tied to the replaced embedding keeps reading the draft's own
        target, draft = _Target(), _toy_draft(tied=True)
        own = _ptr(draft.lm_head.weight)

        def step():
            del draft.model.embed_tokens.weight
            draft.model.embed_tokens.weight = target.model.embed_tokens.weight

        self.assertEqual(wc_draft.share_like_engine(target, draft, step), [])
        self.assertIs(draft.model.embed_tokens.weight, draft.lm_head.weight)
        self.assertEqual(_ptr(draft.lm_head.weight), own)
        # So does a tensor a module keeps in a plain attribute
        target, draft = _Target(), _toy_draft()
        draft.layer.head_ref = [draft.lm_head.weight]
        own = _ptr(draft.lm_head.weight)

        def step2():
            del draft.lm_head.weight
            draft.lm_head.weight = target.lm_head.weight

        self.assertEqual(wc_draft.share_like_engine(target, draft, step2), [])
        self.assertEqual(_ptr(draft.lm_head.weight), own)
        # However deep: an object holding a dict of lists
        target, draft = _Target(), _toy_draft()
        draft.layer.cache = types.SimpleNamespace(heads={"h": [draft.lm_head.weight]})
        own = _ptr(draft.lm_head.weight)
        self.assertEqual(wc_draft.share_like_engine(target, draft, step2), [])
        self.assertEqual(_ptr(draft.lm_head.weight), own)

    def test_new_tensors_other_shapes_and_failures_share_nothing(self):
        target = _Target()
        for case in ("clone", "shape", "raise"):
            with self.subTest(case=case):
                draft = _toy_draft(vocab=6 if case == "shape" else 8)
                before = self._state(draft)
                own = _ptr(draft.lm_head.weight)

                def step():
                    del draft.lm_head.weight
                    if case == "clone":  # a token map narrows a copy of the head
                        draft.lm_head.weight = nn.Parameter(
                            target.lm_head.weight.detach()[:4].clone()
                        )
                    else:
                        draft.lm_head.weight = target.lm_head.weight
                    if case == "raise":
                        raise RuntimeError("boom")

                self.assertEqual(wc_draft.share_like_engine(target, draft, step), [])
                self.assertEqual(self._state(draft), before)
                self.assertEqual(_ptr(draft.lm_head.weight), own)


class TestRealDraftClasses(CustomTestCase):
    """The step the daemon runs is EagleDraftWorker's own init_token_map and
    init_lm_head (DraftPlan.share_step), on the real draft classes. Each is
    built bare with the submodules their methods touch."""

    def _bare(self, cls, **attrs):
        model = object.__new__(cls)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(8, 4)
        model.lm_head = nn.Linear(4, 8, bias=False)
        for k, v in attrs.items():
            setattr(model, k, v)
        return model

    def _share(self, draft, algo="EAGLE", token_map=None):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        target = _Target()
        plan = wc_draft.DraftPlan(SpeculativeAlgorithm.from_string(algo))
        step = plan.share_step(
            target_model=target,
            draft_model=draft,
            target_model_config=types.SimpleNamespace(model_path="", revision=None),
            target_load_config=LoadConfig(),
        )
        modules = [(n, id(m)) for n, m in draft.named_modules()]
        kwargs = {"speculative_token_map": "map.pt"} if token_map is not None else {}
        with (
            _published(speculative_algorithm=algo, **kwargs),
            _no_device_sync(),
            mock.patch(
                "sglang.srt.speculative.eagle_worker_v2.load_token_map",
                return_value=token_map,
            ),
        ):
            shared = wc_draft.share_like_engine(target, draft, step)
        self.assertEqual([(n, id(m)) for n, m in draft.named_modules()], modules)
        for name in shared:
            t = draft.get_parameter(name)
            self.assertIn(
                _ptr(t),
                {_ptr(target.model.embed_tokens.weight), _ptr(target.lm_head.weight)},
            )
        return shared

    def test_real_draft_classes(self):
        from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
        from sglang.srt.models.llama_eagle import LlamaForCausalLMEagle
        from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
        from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

        both = ["lm_head.weight", "model.embed_tokens.weight"]
        for cls in (DeepseekV3ForCausalLMNextN, LlamaForCausalLMEagle):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(self._share(self._bare(cls)), both)
                # A token map narrows a copy of the head: only the embedding
                hot = torch.tensor([0, 2, 4])
                self.assertEqual(
                    self._share(self._bare(cls), token_map=hot),
                    ["model.embed_tokens.weight"],
                )
        with self.subTest(cls="Qwen3_5ForCausalLMMTP"):
            config = types.SimpleNamespace(tie_word_embeddings=False)
            # set_embed_and_head, then set_lm_head_from_target takes the module
            self.assertEqual(
                self._share(self._bare(Qwen3_5ForCausalLMMTP, config=config)), both
            )
            tied = self._bare(
                Qwen3_5ForCausalLMMTP,
                config=types.SimpleNamespace(tie_word_embeddings=True),
            )
            tied.lm_head.weight = tied.model.embed_tokens.weight
            self.assertEqual(self._share(tied), [])  # its head reads its own embedding
        with self.subTest(cls="LlamaForCausalLMEagle3"):
            config = types.SimpleNamespace(hidden_size=4, target_hidden_size=4)
            eagle3 = self._bare(
                LlamaForCausalLMEagle3,
                config=config,
                hot_token_id=None,
                load_lm_head_from_target=False,
            )
            self.assertEqual(
                self._share(eagle3, algo="EAGLE3"), ["model.embed_tokens.weight"]
            )
            eagle3 = self._bare(
                LlamaForCausalLMEagle3,
                config=config,
                hot_token_id=None,
                load_lm_head_from_target=True,
            )
            self.assertEqual(self._share(eagle3, algo="EAGLE3"), both)
        with self.subTest(algo="STANDALONE"):
            from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

            plan = wc_draft.DraftPlan(SpeculativeAlgorithm.from_string("STANDALONE"))
            self.assertIsNone(
                plan.share_step(
                    target_model=None, draft_model=None, target_model_config=None, target_load_config=None
                )
            )  # fmt: skip


# ----------------------------------------------------------------------------
# The daemon serves the target and its draft; an engine installs both
# ----------------------------------------------------------------------------


def _rope():
    from sglang.srt.layers.rotary_embedding.factory import get_rope

    return get_rope(
        head_size=8, rotary_dim=8, max_position=64, base=10000, dtype=torch.float32
    )


class _Toy(nn.Module):
    """A target, or a NextN-style draft that loads neither embedding nor head
    and gets the target's after loading (set_embed_and_head). Both ask for
    the same RoPE, so get_rope hands them one module."""

    def __init__(self, layers=1):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 8)
        self.model.layers = nn.ModuleList(
            nn.Linear(8, 8, bias=False) for _ in range(layers)
        )
        self.model.rotary_emb = _rope()
        self.lm_head = nn.Linear(8, 16, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def forward(self, ids):
        h = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            h = layer(h)
        cos_sin = self.model.rotary_emb.cos_sin_cache[: ids.numel()]
        return self.lm_head(h + cos_sin[:, :8])


class _ToyLoader:
    """What DefaultModelLoader does for the toys: build through
    _initialize_model's hook, then load seeded weights (a NextN draft leaves
    its embedding and head as built)."""

    def __init__(self, layers, seed, load_vocab):
        self.layers, self.seed, self.load_vocab = layers, seed, load_vocab

    def load_model(self, *, model_config, device_config):
        model = _Toy(self.layers)
        callback = loader_module._on_model_constructed.get()
        if callback is not None:
            callback(model)
        torch.manual_seed(self.seed)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if self.load_vocab or name.startswith("model.layers"):
                    p.copy_(torch.randn_like(p))
        return model


class _CopyingTransport:
    """torch_ipc's entries, with each storage copied: a client maps what the
    daemon exported, once per storage however many models name it."""

    name = TORCH_IPC_BACKEND

    def prepare_export(self, state_tensors):
        self.storages = {name: t.clone() for name, (t, _) in state_tensors.items()}
        state_tensors.clear()
        return {name: {"shape": [t.numel()], "dtype": "uint8", "name": name} for name, t in self.storages.items()}  # fmt: skip

    def send_fetch_state_response(self, conn, *, config, entries, pid, preloaded_weights_bytes=0, extra=None):  # fmt: skip
        send_msg(conn, {"status": "ok", "config": config, "entries": entries, "pid": pid,
                        "transport_backend": self.name,
                        "preloaded_weights_bytes": preloaded_weights_bytes, **(extra or {})})  # fmt: skip


class _ClientTransport:
    name = TORCH_IPC_BACKEND

    def __init__(self, storages):
        self.storages, self.imported = storages, []

    def import_tensor(self, entry):
        self.imported.append(entry["name"])
        return self.storages[entry["name"]]

    def close(self):
        pass


_CONFIG = dict(
    model_path="/models/toy", model_arch="Toy", tp_size=1, tp_rank=0, pp_size=1, pp_rank=0,
    dp_size=1, ep_size=1, moe_dp_size=1, moe_dp_rank=0, moe_ep_rank=0, attn_dp_size=1,
    enable_dp_lm_head=False, attn_cp_size=1, moe_dense_tp_size=None, moe_a2a_backend="none",
    quant_method="", quant_config_hash="", dtype="torch.float32", revision="",
    device_capability="", torch_version="",
)  # fmt: skip


class TestServeDraft(CustomTestCase):
    def setUp(self):
        # RotaryEmbedding's CPU construction path (no custom rope op to import)
        self.enterContext(mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True))  # fmt: skip

    def _daemon(self, plan_algo="NEXTN", grow=True):
        """A daemon that went through load()'s steps after the target's load:
        [grow the RoPE caches], describe the target, load the draft, export.
        Returns (daemon, target)."""
        from sglang.srt.layers.rotary_embedding import factory
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.enterContext(mock.patch.dict(factory._ROPE_DICT, clear=True))
        target_loader = _ToyLoader(layers=2, seed=0, load_vocab=True)
        records = {}
        with loader_module.on_model_constructed(
            lambda m: records.__setitem__(id(m), load_diff.record_construction(m))
        ):
            target = target_loader.load_model(model_config=None, device_config=None)
        daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
        daemon.gpu_id, daemon.tp_rank = 0, 0
        daemon.server_args = object()
        daemon.model = target
        daemon.manifest, daemon.draft = None, None
        daemon.state_entries = {}
        daemon.preloaded_weights_bytes = 5
        daemon.transport_name = TORCH_IPC_BACKEND
        daemon.transport_backend = _CopyingTransport()
        daemon.config = CacheConfig(**_CONFIG)
        daemon.load_format = "auto"
        daemon.model_loader_extra_config = {}
        plan = wc_draft.DraftPlan(SpeculativeAlgorithm.from_string(plan_algo))
        draft_config = types.SimpleNamespace(
            model_path="/models/toy",
            hf_config=types.SimpleNamespace(architectures=["ToyNextN"]),
        )
        with (
            mock.patch.object(daemon_module.current_platform, "get_device", return_value=torch.device("cpu")),
            mock.patch("sglang.srt.configs.model_config.ModelConfig.from_server_args", return_value=draft_config),
            mock.patch.object(wc_draft.DraftPlan, "layout", return_value=None),
            mock.patch.object(wc_draft.DraftPlan, "scope", return_value=contextlib.nullcontext()),
            mock.patch.object(ipc_loader.WeightCacheModelLoader, "_resolve_engine_quant", return_value=("", None)),
            mock.patch.object(ipc_loader.WeightCacheModelLoader, "cache_config",
                              return_value=CacheConfig(**{**_CONFIG, "model_arch": "ToyNextN"})),
            mock.patch("sglang.srt.model_loader.loader.get_model_loader",
                       return_value=_ToyLoader(layers=1, seed=1, load_vocab=False)),
            _no_device_sync(),
            _published(speculative_algorithm=plan_algo),
        ):  # fmt: skip
            from sglang.srt.model_executor.model_runner_components.load_model_utils import (
                reserve_rope_cache_for_long_sequences,
            )

            if grow:
                reserve_rope_cache_for_long_sequences(
                    target, types.SimpleNamespace(context_len=64)
                )
            state_tensors = daemon._snapshot(records.pop(id(target)))
            daemon._load_draft(
                plan,
                state_tensors,
                target_model_config=types.SimpleNamespace(
                    model_path="", revision=None, context_len=64
                ),
                target_load_config=LoadConfig(),
                memory_before=0,
            )
            daemon._export_state(state_tensors)
        return daemon, target

    def _fetch(self, daemon, role):
        client, server = socket.socketpair()
        with client, server:
            client.settimeout(5)
            server.settimeout(5)
            thread = threading.Thread(target=daemon._handle_connection, args=(server,))
            thread.start()
            handshake_with_daemon(client)
            request = {"type": "fetch_state", "protocol_version": PROTOCOL_VERSION, "config": dict(_CONFIG)}  # fmt: skip
            if role is not None:
                request["role"] = role
            if role == DRAFT_ROLE:
                request["config"]["model_arch"] = "ToyNextN"
            send_msg(client, request)
            response = recv_msg(client)
            thread.join(5)
        return response

    def _install(self, daemon, response, role, transport):
        loader = ipc_loader.WeightCacheModelLoader(load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE))  # fmt: skip
        loader._transport_backend = transport
        engine = CacheConfig(**{**_CONFIG, **({"model_arch": "ToyNextN"} if role == DRAFT_ROLE else {})})  # fmt: skip
        loader._check_response(response, engine, role)
        with torch.device("meta"):
            skeleton = _Toy(2 if role == TARGET_ROLE else 1)
        with mock.patch.object(ipc_loader, "_initialize_model", return_value=skeleton):
            model = loader._install(
                types.SimpleNamespace(dtype=torch.float32),
                types.SimpleNamespace(device_type="cpu", gpu_id=-1),
                response,
                None,
            )
        ipc_loader._LOADED_MODEL_TRANSPORTS.remove(transport)
        return model

    def test_engine_installs_the_target_and_the_draft(self):
        from sglang.srt.layers.rotary_embedding import factory
        from sglang.srt.model_executor.model_runner_components.load_model_utils import (
            reserve_rope_cache_for_long_sequences,
        )

        daemon, target = self._daemon()
        draft = daemon.draft
        # The daemon grew the shared RoPE cache before it described the target
        grown = target.model.rotary_emb.cos_sin_cache.shape[0]
        self.assertGreater(grown, 64)
        # The draft's embedding and head are the target's storages: exported once
        own = {n for n in draft.names.values() if n.startswith(f"{DRAFT_ROLE}/")}
        self.assertEqual(len(own), 1)  # its one layer
        self.assertEqual(
            set(daemon.transport_backend.storages), set(daemon.state_entries) | own
        )
        self.assertEqual(len(draft.names) - len(own), 3)  # embed, head, RoPE cache

        target_response = self._fetch(daemon, None)  # an older engine names no role
        draft_response = self._fetch(daemon, DRAFT_ROLE)
        self.assertEqual(target_response["role"], TARGET_ROLE)
        self.assertEqual(draft_response["role"], DRAFT_ROLE)
        self.assertEqual(
            draft_response["preloaded_weights_bytes"], draft.preloaded_weights_bytes
        )

        # An engine: the target, its runner grows the RoPE caches (nothing to
        # grow now), then the draft and the worker's step
        transport = _ClientTransport(daemon.transport_backend.storages)
        self.enterContext(mock.patch.dict(factory._ROPE_DICT, clear=True))
        with _published(speculative_algorithm="NEXTN"):
            engine_target = self._install(
                daemon, target_response, TARGET_ROLE, transport
            )
            cache = engine_target.model.rotary_emb.cos_sin_cache
            reserve_rope_cache_for_long_sequences(
                engine_target, types.SimpleNamespace(context_len=64)
            )
            self.assertIs(engine_target.model.rotary_emb.cos_sin_cache, cache)
            engine_draft = self._install(daemon, draft_response, DRAFT_ROLE, transport)
        self.assertIs(engine_draft.model.rotary_emb, engine_target.model.rotary_emb)
        self.assertEqual(engine_target.model.rotary_emb.cos_sin_cache.shape[0], grown)
        # Every storage was mapped by name, the shared ones once per model
        self.assertEqual(
            sorted(set(transport.imported)), sorted(daemon.transport_backend.storages)
        )
        self.assertEqual(_ptr(engine_draft.model.embed_tokens.weight), _ptr(engine_target.model.embed_tokens.weight))  # fmt: skip
        engine_draft.set_embed_and_head(*engine_target.get_embed_and_head())
        ids = torch.arange(4)
        torch.testing.assert_close(engine_target(ids), target(ids))
        daemon_draft = draft.model
        daemon_draft.set_embed_and_head(*target.get_embed_and_head())
        torch.testing.assert_close(engine_draft(ids), daemon_draft(ids))

    def test_without_the_daemons_rope_growth_the_draft_is_refused(self):
        # The engine's target runner grows the shared RoPE cache; a draft the
        # daemon built on the short one is another structure, and installing
        # it would put the short cache back under the target.
        from sglang.srt.layers.rotary_embedding import factory
        from sglang.srt.model_executor.model_runner_components.load_model_utils import (
            reserve_rope_cache_for_long_sequences,
        )

        daemon, _ = self._daemon(grow=False)
        transport = _ClientTransport(daemon.transport_backend.storages)
        self.enterContext(mock.patch.dict(factory._ROPE_DICT, clear=True))
        with _published(speculative_algorithm="NEXTN"):
            engine_target = self._install(
                daemon, self._fetch(daemon, None), TARGET_ROLE, transport
            )
            reserve_rope_cache_for_long_sequences(
                engine_target, types.SimpleNamespace(context_len=64)
            )
            grown = engine_target.model.rotary_emb.cos_sin_cache
            self.assertGreater(grown.shape[0], 64)
            with self.assertRaises(load_diff.SnapshotError):
                self._install(
                    daemon, self._fetch(daemon, DRAFT_ROLE), DRAFT_ROLE, transport
                )
        self.assertIs(engine_target.model.rotary_emb.cos_sin_cache, grown)

    def test_a_standalone_draft_shares_nothing_but_the_rope_cache(self):
        daemon, _ = self._daemon("STANDALONE")
        shared = [
            n for n in daemon.draft.names.values() if not n.startswith(f"{DRAFT_ROLE}/")
        ]
        self.assertEqual(len(shared), 1)  # get_rope's one module

    def test_roles_the_daemon_does_not_serve(self):
        daemon, _ = self._daemon()
        response = self._fetch(daemon, "draft:1")
        self.assertEqual(response["status"], "error")
        self.assertIn("draft:1", response["message"])
        daemon.draft = None
        response = self._fetch(daemon, DRAFT_ROLE)
        self.assertEqual(response["status"], "error")
        self.assertIn("speculative decoding settings", response["message"])
        with self.assertRaisesRegex(RuntimeError, "refused the draft model"):
            ipc_loader.WeightCacheModelLoader._check_response(
                response, None, DRAFT_ROLE
            )

    def test_a_daemon_from_before_roles(self):
        # It answers a draft request with its target (and no role)
        daemon, _ = self._daemon()
        response = {k: v for k, v in self._fetch(daemon, None).items() if k != "role"}
        engine = CacheConfig(**_CONFIG)
        ipc_loader.WeightCacheModelLoader._check_response(response, engine, TARGET_ROLE)
        with self.assertRaisesRegex(
            RuntimeError, "does not serve speculative decoding drafts"
        ):
            ipc_loader.WeightCacheModelLoader._check_response(
                response, engine, DRAFT_ROLE
            )

    def test_the_draft_stamp_must_agree(self):
        daemon, _ = self._daemon()
        response = self._fetch(daemon, DRAFT_ROLE)
        self.assertEqual(response["draft_stamp"], daemon.draft.stamp)
        self.assertNotIn("draft_stamp", self._fetch(daemon, None))
        check = ipc_loader.WeightCacheModelLoader._check_draft_stamp
        with _published(speculative_algorithm="NEXTN"):
            check(response["draft_stamp"])
            for other in (
                {**response["draft_stamp"], "moe_runner_backend": "triton"},
                {**response["draft_stamp"], "speculative_algorithm": "STANDALONE"},
                None,  # a daemon that sends none
            ):
                with self.subTest(other=other):
                    with self.assertRaisesRegex(RuntimeError, "other settings"):
                        check(other)

    def test_a_draft_config_mismatch_is_reported_as_one(self):
        daemon, _ = self._daemon()
        client, server = socket.socketpair()
        with client, server:
            client.settimeout(5)
            server.settimeout(5)
            thread = threading.Thread(target=daemon._handle_connection, args=(server,))
            thread.start()
            handshake_with_daemon(client)
            config = {**_CONFIG, "model_arch": "ToyNextN", "tp_size": 2}
            send_msg(
                client,
                {
                    "type": "fetch_state",
                    "protocol_version": PROTOCOL_VERSION,
                    "config": config,
                    "role": DRAFT_ROLE,
                },
            )
            response = recv_msg(client)
            thread.join(5)
        self.assertEqual(response["status"], "mismatch")
        with self.assertRaisesRegex(RuntimeError, "config mismatch"):
            ipc_loader.WeightCacheModelLoader._check_response(
                response, CacheConfig(**config), DRAFT_ROLE
            )


class TestMoeFlagsLikeTheEngine(CustomTestCase):
    """The engine's scheduler seeds the MoE flags only for a target with MoE
    layers; the daemon does the same, so a draft built after a dense target
    sees the flags an engine's draft runner sees (draft_stamp compares them)."""

    def test_which_configs_have_moe_layers(self):
        from sglang.srt.layers.moe.utils import has_moe_layers

        for config, moe in (
            (types.SimpleNamespace(hidden_size=8), False),  # Qwen3-8B
            (types.SimpleNamespace(num_experts_per_tok=8), True),
            (types.SimpleNamespace(top_k_experts=2), True),  # Gemma4
            (types.SimpleNamespace(moe_topk=4), True),  # LongCat-2.0
        ):
            with self.subTest(config=vars(config)):
                self.assertEqual(has_moe_layers(config), moe)

    def test_scheduler_and_daemon_seed_alike(self):
        # Read from the sources: importing the scheduler needs the GPU stack
        import importlib.util

        seeded = "if has_moe_layers({}.hf_text_config):\n{}initialize_moe_config()"
        for module, holder, indent in (
            ("sglang.srt.managers.scheduler", "self.model_config", " " * 12),
            ("sglang.srt.weight_cache.daemon", "model_config", " " * 12),
        ):
            with open(importlib.util.find_spec(module).origin) as f:
                src = f.read()
            with self.subTest(module=module):
                self.assertIn(seeded.format(holder, indent), src)
                self.assertEqual(src.count("initialize_moe_config()"), 1)

    def test_an_unseeded_flag_reads_as_its_default(self):
        from sglang.srt.layers.moe import initialize_moe_config

        with _published(speculative_algorithm="EAGLE", moe_runner_backend="triton"):
            dense = wc_draft.draft_stamp()  # a dense target's process
            initialize_moe_config()
            moe = wc_draft.draft_stamp()
        self.assertEqual(dense["moe_runner_backend"], "auto")
        self.assertEqual(moe["moe_runner_backend"], "triton")


class TestEngineRole(CustomTestCase):
    def test_a_draft_runner_asks_for_the_draft(self):
        loader = ipc_loader.WeightCacheModelLoader(load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE))  # fmt: skip
        seen = []

        def fetch(model_config, device_config):
            seen.append(loader.role)
            return None  # no daemon: client mode loads from disk

        for is_draft, role in ((False, TARGET_ROLE), (True, DRAFT_ROLE)):
            with (
                self.subTest(is_draft=is_draft),
                mock.patch.object(
                    loader, "_resolve_engine_quant", return_value=("", None)
                ),
                mock.patch.object(loader, "_fetch_from_cache", side_effect=fetch),
                mock.patch.object(loader, "_fallback_load", return_value="disk"),
            ):
                loader.role = TARGET_ROLE
                config = types.SimpleNamespace(is_draft_model=is_draft)
                self.assertEqual(
                    loader.load_model(model_config=config, device_config=None), "disk"
                )
                self.assertEqual(seen[-1], role)

    def test_fallback_keeps_the_draft_layers_and_group(self):
        load_config = LoadConfig(load_format=LoadFormat.IPC_CACHE, draft_model_idx=2)
        load_config.load_group = "group"
        loader = ipc_loader.WeightCacheModelLoader(load_config=load_config, fallback_load_format="auto")  # fmt: skip
        fallback = mock.Mock()
        loader._fallback_loader_cls = fallback
        loader._fallback_load(model_config=None, device_config=None)
        used = fallback.call_args.args[0]
        self.assertEqual((used.draft_model_idx, used.load_group), (2, "group"))


if __name__ == "__main__":
    unittest.main()
