# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding drafts in the weight cache.

An engine's draft runner loads its draft like the target: through
WeightCacheModelLoader, from the daemon on its GPU, naming the draft role in
its fetch_state request (protocol.DRAFT_ROLE). So the daemon loads the draft
too, in the same process and after the target, as the engine's draft worker
does (``plan_draft``):

- in the scope the worker builds its draft in: its parallel layout, the
  speculative MoE backends and the draft build scope (``DraftPlan.scope``);
- what loading produced goes to the engine through load_diff, in a manifest of
  its own, which the engine checks like the target's;
- a tensor that the worker gives the draft from the target after loading (the
  EAGLE family's embedding and LM head) is kept once (``share_like_engine``),
  and the daemon exports a storage that the target and the draft share once;
- what the build scope sets beyond CacheConfig (``draft_stamp``) must be the
  same on both sides.

The drafts of EAGLE, EAGLE3 and NEXTN (EagleDraftWorker) and STANDALONE
(StandaloneDraftWorker) are served; NGRAM has no draft model.
``unsupported_reason`` names what is refused, and why.
"""

import contextlib
import dataclasses
import itertools
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

# The algorithms whose draft the daemon builds (NEXTN is EAGLE's alias), and
# the one without a draft model.
_SERVED = ("EAGLE", "EAGLE3", "NEXTN", "STANDALONE")
_NO_DRAFT = ("NGRAM",)


def unsupported_reason(server_args) -> Optional[str]:
    """Why the weight cache cannot serve this speculative decoding setup
    (None: it can). Checked when the server arguments are resolved and again
    by the daemon."""
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm is None:
        return None
    name = cfg.speculative_algorithm.upper()
    if name in _NO_DRAFT:
        return None
    if name not in _SERVED:
        return (
            f"--speculative-algorithm {name}: the daemon builds the drafts of "
            f"{', '.join(_SERVED)} only"
        )
    if name != "STANDALONE" and cfg.enable_multi_layer_eagle:
        return (
            "--enable-multi-layer-eagle: the daemon builds one draft model, not "
            "one per step"
        )
    if cfg.pp_size > 1:
        return (
            "speculative decoding with --pp-size > 1: a draft on the last stage "
            "reads the target's embedding from checkpoint files, which the "
            "daemon's transport does not provide"
        )
    if cfg.nnodes > 1:
        return "speculative decoding with --nnodes > 1: not tested yet"
    return None


@dataclasses.dataclass
class DraftPlan:
    """How the engine's draft worker for ``algorithm`` builds its draft."""

    algorithm: Any  # SpeculativeAlgorithm

    @property
    def standalone(self) -> bool:
        return self.algorithm.is_standalone()

    def layout(self, server_args, target_model_config):
        """The draft's parallel layout, resolved in the target's scope as the
        worker resolves it (EagleDraftWorker / StandaloneDraftWorker.__init__)."""
        from sglang.srt.configs.draft_parallel import (
            DraftParallelSpec,
            DraftTPExecution,
        )
        from sglang.srt.speculative.draft_parallel import (
            model_draft_parallel_spec,
            resolve_draft_parallel_layout,
        )

        if self.standalone:
            return resolve_draft_parallel_layout(
                DraftParallelSpec(tp_execution=DraftTPExecution.TARGET_LAYOUT)
            )
        from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
            resolve_spec_aux_hidden_state_config,
        )

        # What the target runner's spec_aux_config holds and the worker reads
        aux = resolve_spec_aux_hidden_state_config(
            server_args=server_args,
            model_config=target_model_config,
            spec_algorithm=self.algorithm,
            is_draft_worker=False,
        )
        return resolve_draft_parallel_layout(
            model_draft_parallel_spec(
                aux.draft_model_config, spec_algorithm=self.algorithm
            )
        )

    @contextlib.contextmanager
    def scope(self, layout):
        """The contexts the worker builds and loads its draft in, in the same
        order (EagleDraftWorker / StandaloneDraftWorker.__init__)."""
        from sglang.srt.layers.moe.utils import (
            draft_model_build_scope,
            speculative_moe_a2a_backend_context,
            speculative_moe_backend_context,
        )
        from sglang.srt.speculative.spec_utils import (
            draft_pp_context,
            draft_tp_context,
        )

        if self.standalone:
            contexts = (
                draft_tp_context(layout),
                draft_pp_context(),
                draft_model_build_scope(),
            )
        else:
            contexts = (
                draft_tp_context(layout),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                draft_model_build_scope(),
            )
        with contextlib.ExitStack() as stack:
            for context in contexts:
                stack.enter_context(context)
            yield

    def share_step(
        self, *, target_model, draft_model, target_model_config, target_load_config
    ) -> Optional[Callable[[], Any]]:
        """The step the worker runs on the loaded draft that gives it the
        target's tensors (None: it gives none). It is the worker's own code,
        EagleDraftWorker.init_token_map and init_lm_head, run on these
        models."""
        if self.standalone:
            return None  # StandaloneDraftWorker.init_lm_head shares nothing
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.speculative_algorithm = self.algorithm
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=target_model,
                model_config=target_model_config,
                load_config=target_load_config,
            )
        )
        worker.draft_runner = SimpleNamespace(model=draft_model)

        def step():
            worker.init_token_map()
            worker.init_lm_head()

        return step


def plan_draft(server_args) -> Optional[DraftPlan]:
    """The draft the engine's draft runner fetches from the daemon, as
    ``SpeculativeAlgorithm.create_worker`` picks the worker; None when there
    is no draft model. Raises ValueError for a setup the daemon cannot
    serve."""
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    reason = unsupported_reason(server_args)
    if reason is not None:
        raise ValueError(f"The weight cache does not support {reason}.")
    name = resolving_view(server_args).speculative_algorithm
    if name is None or name.upper() in _NO_DRAFT:
        return None
    algorithm = SpeculativeAlgorithm.from_string(name)
    if algorithm.get_platform_worker_cls() is not None:
        raise ValueError(
            f"The weight cache does not support --speculative-algorithm {name} "
            f"on this platform: its own worker builds the draft."
        )
    return DraftPlan(algorithm=algorithm)


def _value_name(value) -> str:
    return str(getattr(value, "value", value))


def draft_stamp() -> Dict[str, str]:
    """What shapes a draft's modules and post-processed weights beyond its
    CacheConfig: the speculative algorithm and the MoE backends in effect where
    it is built (a flag its process never set counts as what its readers take
    it for). The daemon sends the one taken in its draft's scope; an engine's
    draft runner, which loads in its worker's scope, compares its own."""
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_flags, get_spec

    moe = get_flags().moe
    runner, a2a = moe.runner_backend, moe.a2a_backend
    return {
        "speculative_algorithm": _value_name(get_spec().speculative_algorithm),
        "moe_runner_backend": _value_name(
            MoeRunnerBackend.AUTO if runner is None else runner
        ),
        "moe_a2a_backend": _value_name(MoeA2ABackend.NONE if a2a is None else a2a),
    }


def _strided(t) -> bool:
    return isinstance(t, torch.Tensor) and not t.is_meta and t.layout == torch.strided


def _held_tensor_ids(model: nn.Module) -> set:
    """Every tensor a model holds anywhere load-diff looks: parameters,
    buffers, plain attributes and the containers and objects under them."""
    from .load_diff_format import walk_model

    walk = walk_model(model)
    return {id(t) for t in (*walk.tensors.values(), *walk.local.values())}


def _saved_tensor(saved: Dict[int, tuple], module: nn.Module, name: str):
    """The tensor ``module`` held at ``name`` before the share step."""
    *path, leaf = name.split(".")
    for part in path:
        entry = saved.get(id(module))
        if entry is None:
            return None
        module = entry[2].get(part)
    entry = saved.get(id(module))
    if entry is None:
        return None
    return entry[0].get(leaf, entry[1].get(leaf))


def share_like_engine(
    target: nn.Module, draft: nn.Module, step: Callable[[], Any], tag: str = ""
) -> List[str]:
    """Keep one copy of what a draft takes from its target.

    ``step`` is the worker's own step (an engine runs it after loading the
    draft) that puts target tensors or modules in the draft. Here it runs on
    the daemon's models and is undone, except that every draft tensor it
    replaced everywhere (the draft no longer holds it afterwards) by a target
    tensor of the same shape, dtype and strides now holds that tensor's data.
    The draft's own copy is freed, and the draft's export names the target's
    storage, which an engine maps once for both. The engine installs the draft
    as loaded and runs the step itself, as without the cache, so a tensor the
    step leaves in the draft (an LM head tied to a replaced embedding, say)
    keeps its own data. Returns the names of the draft tensors that hold
    target data.
    """
    target_storages = {
        t.untyped_storage().data_ptr()
        for t in itertools.chain(target.parameters(), target.buffers())
        if _strided(t)
    }
    target_modules = {id(m) for m in target.modules()}
    # Every draft module as loaded: its tensors, children and attributes
    saved: Dict[int, tuple] = {}
    for m in draft.modules():
        saved[id(m)] = (
            dict(m._parameters),
            dict(m._buffers),
            dict(m._modules),
            dict(vars(m)),
            m,
        )
    names: Dict[int, str] = {}
    for name, t in itertools.chain(
        draft.named_parameters(remove_duplicate=False),
        draft.named_buffers(remove_duplicate=False),
    ):
        names.setdefault(id(t), name)

    def from_target(t) -> bool:
        return _strided(t) and t.untyped_storage().data_ptr() in target_storages

    pairs = []  # (draft tensor as loaded, the target tensor the step put there)
    try:
        step()
        still_held = _held_tensor_ids(draft)
        for params, buffers, children, _, m in saved.values():
            for store, before in ((m._parameters, params), (m._buffers, buffers)):
                for leaf, old in before.items():
                    new = store.get(leaf)
                    if old is not None and new is not old and from_target(new):
                        pairs.append((old, new))
            for leaf, old_child in children.items():
                new_child = m._modules.get(leaf)
                if (
                    old_child is None
                    or new_child is old_child
                    or id(new_child) not in target_modules
                ):
                    continue
                # A module taken whole (set_lm_head_from_target): by tensor name
                for name, new in itertools.chain(
                    new_child.named_parameters(remove_duplicate=False),
                    new_child.named_buffers(remove_duplicate=False),
                ):
                    old = _saved_tensor(saved, old_child, name)
                    if old is not None and old is not new and from_target(new):
                        pairs.append((old, new))
        pairs = [(old, new) for old, new in pairs if id(old) not in still_held]
    except Exception as e:
        logger.warning(
            f"{tag} could not run the draft worker's step that gives the draft "
            f"the target's tensors ({type(e).__name__}: {e}); the draft keeps "
            f"its own copies of them."
        )
        pairs = []
    finally:
        for params, buffers, children, attrs, m in saved.values():
            for store, before in (
                (m._parameters, params),
                (m._buffers, buffers),
                (m._modules, children),
            ):
                store.clear()
                store.update(before)
            current = vars(m)
            for k in [k for k in current if k not in attrs]:
                del current[k]
            for k, v in attrs.items():
                if current.get(k, attrs) is not v:
                    current[k] = v
    shared, done = [], set()
    for old, new in pairs:
        if id(old) in done or not _strided(old):
            continue
        done.add(id(old))
        if (old.shape, old.dtype, old.device, old.stride()) != (
            new.shape,
            new.dtype,
            new.device,
            new.stride(),
        ):
            continue
        # The loaded tensor object keeps its Python attributes; only its data
        # is now the target's.
        old.data = new.detach()
        shared.append(names.get(id(old), "?"))
    return sorted(shared)
