"""Internal storage backing ``SpeculativeAlgorithm.register``. Plugins
should use that classmethod API; do not import from this module directly.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

import torch

from sglang.srt.arg_groups.overrides import resolving_view

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.verify_extension import (
        LinearAttnStateLayout,
        LinearAttnVerifyExtension,
    )
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.runner.decode_graph_variants import (
        DecodeGraphRunnerInfo,
        DecodeGraphVariants,
    )
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpecInput

WorkerFactory = Callable[["ServerArgs"], Type]
ServerArgsValidator = Callable[["ServerArgs"], None]

logger = logging.getLogger(__name__)


class CustomSpecAlgo:
    """A plugin-registered speculative algorithm. Duck-types
    ``SpeculativeAlgorithm`` enum values (same ``is_*()`` / ``create_worker``
    interface).

    Plugins may subclass this to override any ``is_*()`` / ``supports_*()`` /
    ``create_worker`` method (e.g. to integrate with builtin-specific
    branches like ``if spec_algorithm.is_eagle():`` in scheduler /
    model_runner). Pass the subclass via ``spec_class=...`` at registration.

    Defaults: all ``is_*()`` return ``False`` except ``is_speculative``.

    ``supports_overlap=False`` is deprecated: the spec V1 worker path has been
    removed, so such algorithms run on the V2 scheduler schema with overlap
    disabled (synchronous). Migrate plugin workers to the V2 schema and
    overlap scheduling.
    """

    def __init__(
        self,
        name: str,
        factory: WorkerFactory,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
    ):
        self.name = name
        self.factory = factory
        self.supports_overlap = supports_overlap
        self.validate_server_args = validate_server_args

    def __repr__(self) -> str:
        return f"CustomSpecAlgo({self.name!r})"

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def is_speculative(self) -> bool:
        return True

    def is_eagle(self) -> bool:
        return False

    def supports_mixed_chunk(self) -> bool:
        return False

    def is_eagle3(self) -> bool:
        return False

    def uses_aux_hidden_states(self) -> bool:
        """Whether the target captures auxiliary hidden states for the draft,
        from the layers the draft's ``eagle_config`` names, and the draft reads
        their concatenation. Lets an EAGLE-family plugin get only that
        EAGLE-3 behavior without answering ``is_eagle3()``."""
        return False

    def is_frozen_kv_mtp(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False

    def is_uno(self) -> bool:
        return False

    def is_dspark(self) -> bool:
        return False

    def is_dflash_family(self) -> bool:
        return False

    def is_standalone(self) -> bool:
        return False

    def is_ngram(self) -> bool:
        return False

    def supports_target_verify_for_draft(self) -> bool:
        return False

    def supports_prefill_shared_read_done(self) -> bool:
        # Whether target EXTEND has no later speculative shared-buffer reader.
        return False

    def supports_ragged_verify(self) -> bool:
        return False

    def supports_grammar_overlap(self) -> bool:
        # Whether the worker advances the grammar FSM inside verify() (via the
        # scheduler's grammar barrier), letting spec + grammar decode overlap.
        return False

    def supports_mm_inputs(self) -> bool:
        """Whether requests with multimodal inputs may be served. ``False``
        also switches the server warmup to a text-only request."""
        return True

    def validate_request(
        self, req: Req, *, has_mm_inputs: bool, enable_overlap: bool
    ) -> Optional[str]:
        """Return why this algorithm cannot serve ``req``, or ``None``.

        Called by the scheduler on admission, before the request is queued; a
        message aborts the request with it. ``has_mm_inputs`` says whether the
        request carries images / audio / video (``req`` does not hold them
        yet). Overrides should call ``super().validate_request(...)`` to keep
        the ``supports_mm_inputs()`` check.
        """
        if has_mm_inputs and not self.supports_mm_inputs():
            return (
                f"Speculative algorithm {self.name} does not support multimodal inputs."
            )
        if self.name == "EAGLE_HARNESS":
            # The 0.1.1 plugin's fused tail does not consult custom processors
            # or watermark state. Reject at admission, before it can select
            # the greedy token-only graph for such a request.
            if getattr(req, "custom_logit_processor", None) is not None:
                return "EAGLE-4 does not support custom logit processors."
            if getattr(getattr(req, "sampling_params", None), "watermark", None):
                return "EAGLE-4 does not support watermarking."
        return None

    def has_draft_kv(self) -> bool:
        # Conservative default: the larger KV reserve.
        return True

    def need_topk(self) -> bool:
        """Whether the decode relay carries the draft top-k (``topk_p`` /
        ``topk_index``). Follows ``is_eagle()`` like the builtins."""
        return self.is_eagle() or self.is_standalone()

    def carries_draft_hidden_states(self) -> bool:
        """Whether the disagg prefill->decode transfer carries draft hidden
        states. Follows ``is_eagle()`` like the builtins."""
        return self.is_eagle()

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ) -> FutureMap:
        """The scheduler's step-to-step relay (always on, overlap or not)."""
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device,
            self,
            req_to_token_pool,
            needs_cpu_seq_lens,
            needs_confidence_relay,
        )

    def handle_server_args(self, server_args: ServerArgs) -> None:
        pass

    def req_to_token_extra_len(self, page_size: int) -> int:
        """Minimum ``req_to_token`` row headroom (tokens) past the context
        length that this algorithm writes into, e.g. draft / verify slots
        allocated past ``seq_len``. The row is sized to the max of this and
        the default headroom, so returning 0 (the default) keeps it."""
        return 0

    def create_decode_graph_variants(
        self, info: DecodeGraphRunnerInfo
    ) -> Optional[DecodeGraphVariants]:
        """Extra CUDA-graph variants for a decode / target-verify runner
        (see ``model_executor/runner/decode_graph_variants.py``), for example
        one target-verify graph per verify width. Called once per runner,
        target and draft alike, before the runner captures; ``info`` says
        which (``is_draft_worker``, ``capture_forward_mode``), the widest
        verify it fits and its device. ``None`` (the default) keeps one graph
        per batch size at the static width."""
        return None

    def create_linear_attn_verify_extension(
        self, state: LinearAttnStateLayout
    ) -> Optional[LinearAttnVerifyExtension]:
        """Target-verify extension for a GDN or Mamba2 linear-attention
        backend with this ``state`` (``state.kind``; see
        ``layers/attention/linear/verify_extension.py``): replace the verify
        conv1d and recurrence with your own kernels, for example a verify over
        a non-chain draft layout. Called once per backend. ``None`` (the
        default) keeps the backend's kernels."""
        return None

    def resolve_max_speculative_num_draft_tokens(
        self, server_args: ServerArgs
    ) -> Optional[int]:
        """Return the largest draft-token width this algorithm may use.

        The default covers static algorithms and adaptive algorithms whose
        runtime states never exceed their startup width. Overrides must not
        return less than ``server_args.speculative_num_draft_tokens``.
        """
        from sglang.srt.arg_groups.overrides import resolving_view

        return resolving_view(server_args).speculative_num_draft_tokens

    def create_worker(self, server_args: ServerArgs) -> Type:

        cfg = resolving_view(server_args)
        # eagle-harness 0.1.1 builds its own CUDA/HIP draft and verify kernels.
        # It cannot use the builtin EAGLE NPU/CPU kernels as a fallback.
        if self.name == "EAGLE_HARNESS" and cfg.device != "cuda":
            raise NotImplementedError(
                "EAGLE-4 (EAGLE_HARNESS) requires CUDA or ROCm; "
                f"device={cfg.device!r} is not supported by eagle-harness 0.1.1."
            )
        if not cfg.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
        if not self.supports_overlap:
            # Reached only when overlap is disabled, so the algorithm really
            # does run synchronously on the V2 schema below.
            logger.warning(
                "Speculative algorithm %s is registered with "
                "supports_overlap=False, which is deprecated: the spec V1 "
                "worker path has been removed, and the algorithm now runs on "
                "the V2 scheduler schema with overlap disabled (synchronous). "
                "Migrate the plugin worker to support overlap scheduling.",
                self.name,
            )
        return self.factory(server_args)

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        return num_draft_tokens

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        """Build the disaggregation draft input for ``batch``, or ``None``.

        The speculative config comes from ``runtime_context.get_spec()``, which
        follows a runtime override where the startup record does not.
        """
        return None


_REGISTRY: Dict[str, CustomSpecAlgo] = {}

# CLI spellings that are not ``SpeculativeAlgorithm`` members but still resolve
# to a builtin (e.g. NEXTN -> EAGLE). Reserved alongside the enum members so
# plugins cannot shadow them.
_RESERVED_ALIASES = frozenset({"NEXTN"})


def _reserved_names() -> frozenset:
    """Names plugins cannot register under: every ``SpeculativeAlgorithm``
    member plus ``_RESERVED_ALIASES``.

    Derived from the enum (lazily, to avoid a circular import — ``spec_info``
    imports this module) so any new builtin is reserved automatically without
    editing a second list.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return frozenset(algo.name for algo in SpeculativeAlgorithm) | _RESERVED_ALIASES


def _assert_custom_spec_algo_conforms(spec_class: Type[CustomSpecAlgo]) -> None:
    """Fail fast if ``spec_class`` drifts from the ``SpeculativeAlgorithm``
    duck-typing contract (every public instance method of the enum).

    ``from_string`` returns either type and callers dispatch on the shared
    ``is_*()`` / ``supports_*()`` interface without isinstance checks, so every
    such method on the enum must also exist on the registered spec class —
    otherwise a plugin-registered algo hits ``AttributeError`` at a call site
    (this is how ``is_some`` / ``is_frozen_kv_mtp`` silently went missing). New
    predicates are covered automatically; no second list to maintain.

    Called from ``register_algorithm`` rather than at import time because
    ``spec_info`` imports this module, so ``SpeculativeAlgorithm`` does not yet
    exist while this module is loading; at registration time it is fully
    defined.
    """
    # NOTE: use ``vars()`` not ``dir()`` for the enum — ``EnumMeta.__dir__``
    # hides instance methods, so ``dir(SpeculativeAlgorithm)`` would yield an
    # empty interface and turn this guard into a silent no-op.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # Every public instance method: the scheduler and model runner call more
    # than the predicates (e.g. ``create_future_map`` at scheduler init).
    interface = {
        name
        for name, value in vars(SpeculativeAlgorithm).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    missing = sorted(interface - set(dir(spec_class)))
    if missing:
        raise TypeError(
            f"{spec_class.__name__} is missing duck-typed methods from "
            f"SpeculativeAlgorithm: {missing}. Add them to {spec_class.__name__} "
            "so plugin-registered algorithms stay dispatchable."
        )


def register_algorithm(
    name: str,
    *,
    supports_overlap: bool = False,
    validate_server_args: Optional[ServerArgsValidator] = None,
    spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
) -> Callable[[WorkerFactory], WorkerFactory]:
    """Return a decorator that registers a plugin algorithm under ``name``.

    Pass a ``spec_class`` subclass of ``CustomSpecAlgo`` to override any
    ``is_*()`` / ``supports_*()`` / ``create_worker`` method.
    """
    upper = name.upper()
    if upper in _reserved_names():
        raise ValueError(
            f"'{upper}' is a reserved speculative algorithm name; cannot be re-registered."
        )
    if upper in _REGISTRY:
        raise ValueError(f"Speculative algorithm '{upper}' already registered.")
    _assert_custom_spec_algo_conforms(spec_class)

    def decorator(factory: WorkerFactory) -> WorkerFactory:
        _REGISTRY[upper] = spec_class(
            name=upper,
            factory=factory,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
        )
        return factory

    return decorator


def get_spec(name: Optional[str]) -> Optional[CustomSpecAlgo]:
    """Return the registered spec for ``name``, or ``None`` for builtin /
    unknown names."""
    if name is None:
        return None
    return _REGISTRY.get(name.upper())
