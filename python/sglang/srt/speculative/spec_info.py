from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    ServerArgsValidator,
    WorkerFactory,
)
from sglang.srt.speculative.spec_registry import get_spec as _get_registered_spec
from sglang.srt.speculative.spec_registry import (
    register_algorithm as _register_algorithm,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker

logger = logging.getLogger(__name__)


class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and
    both expose the same ``is_*()`` / ``create_worker`` interface so callers
    dispatch uniformly without isinstance checks.
    """

    DFLASH = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"Unknown speculative algorithm name: {name}")

    @classmethod
    def register(
        cls,
        name: str,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
        spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
    ) -> Callable[[WorkerFactory], WorkerFactory]:
        """Decorator to register a plugin speculative algorithm. The factory
        takes ``server_args`` and returns the worker class. Pass a
        ``CustomSpecAlgo`` subclass via ``spec_class`` to override any
        ``is_*()`` / ``create_worker`` method.

        Example:
            @SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=False)
            def _factory(server_args):
                return MySpecWorker
        """
        return _register_algorithm(
            name,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
            spec_class=spec_class,
        )

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def is_speculative(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # FIXME(kpham_sgl): Remove FROZEN_KV_MTP here once we
        # have established support for it in the scheduler.
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        )

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_frozen_kv_mtp(self) -> bool:
        return self == SpeculativeAlgorithm.FROZEN_KV_MTP

    def is_dflash(self) -> bool:
        return self == SpeculativeAlgorithm.DFLASH

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def is_decoupled_verify(self) -> bool:
        return False

    def is_decoupled_draft(self) -> bool:
        return False

    def supports_spec_v2(self) -> bool:
        return (self.is_eagle() and not self.is_frozen_kv_mtp()) or self.is_standalone()

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        enable_overlap = not server_args.disable_overlap_schedule

        if self.is_dflash():
            if enable_overlap:
                raise ValueError(
                    "DFLASH does not support overlap scheduling (spec v2)."
                )
            from sglang.srt.speculative.dflash_worker import DFlashWorker

            return DFlashWorker

        if self.is_frozen_kv_mtp():
            if enable_overlap:
                raise ValueError(
                    "FROZEN_KV_MTP does not support spec v2. Disable overlap "
                    "scheduling to use FrozenKVMTPWorker."
                )

            from sglang.srt.speculative.frozen_kv_mtp_worker import (
                FrozenKVMTPWorker,
            )

            return FrozenKVMTPWorker

        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            # FIXME: migrate to EagleWorker
            if enable_overlap:
                from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                    MultiLayerEagleWorkerV2,
                )

                return MultiLayerEagleWorkerV2

            from sglang.srt.speculative.multi_layer_eagle_worker import (
                MultiLayerEagleWorker,
            )

            return MultiLayerEagleWorker

        elif self.is_eagle():
            if enable_overlap:
                from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

                return EAGLEWorkerV2

            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            return EAGLEWorker
        elif self.is_standalone():
            if enable_overlap:
                from sglang.srt.speculative.standalone_worker_v2 import (
                    StandaloneWorkerV2,
                )

                return StandaloneWorkerV2

            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            return StandaloneWorker
        elif self.is_ngram():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


class DecoupledVerifySpecAlgo(CustomSpecAlgo):
    def is_decoupled_verify(self) -> bool:
        return True

    def supports_spec_v2(self) -> bool:
        return False

    @staticmethod
    def validate_server_args(server_args: ServerArgs) -> None:
        if server_args.enable_dp_attention:
            raise ValueError(
                "decoupled speculative decoding does not support dp attention."
            )
        if server_args.dp_size != 1:
            raise ValueError("decoupled verifier currently requires dp_size == 1.")
        if not server_args.decoupled_spec_bind_endpoint:
            raise ValueError(
                "decoupled speculative decoding requires --decoupled-spec-bind-endpoint."
            )
        if not server_args.decoupled_spec_connect_endpoints:
            raise ValueError(
                "decoupled speculative decoding requires a non-empty "
                "--decoupled-spec-connect-endpoints list."
            )
        if server_args.decoupled_spec_rank is None:
            raise ValueError(
                "decoupled speculative decoding requires --decoupled-spec-rank."
            )
        if int(server_args.decoupled_spec_rank) < 0:
            raise ValueError("--decoupled-spec-rank must be non-negative.")
        if server_args.page_size is not None and server_args.page_size > 1:
            raise ValueError(
                "decoupled drafter currently requires page_size == 1 because "
                "token rollback is token-granular."
            )

        if server_args.max_running_requests is None:
            server_args.max_running_requests = 64
            logger.warning(
                "Max running requests is reset to 64 for decoupled speculative decoding. "
                "You can override this by explicitly setting --max-running-requests."
            )

        if not server_args.disable_overlap_schedule:
            server_args.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled for decoupled speculative decoding."
            )

        if server_args.enable_mixed_chunk:
            server_args.enable_mixed_chunk = False
            logger.warning(
                "Mixed chunked prefill is disabled for decoupled speculative decoding."
            )

        if server_args.speculative_num_steps is None:
            raise ValueError(
                "decoupled speculative decoding requires speculative_num_steps to be set."
            )

        if (
            server_args.speculative_eagle_topk is not None
            and server_args.speculative_eagle_topk != 1
        ):
            raise ValueError(
                "decoupled speculative decoding currently only supports "
                "speculative_eagle_topk == 1."
            )
        server_args.speculative_eagle_topk = 1

        expected_num_draft_tokens = int(server_args.speculative_num_steps) + 1
        if server_args.speculative_num_draft_tokens != expected_num_draft_tokens:
            logger.warning(
                "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 "
                "for decoupled speculative decoding."
            )
            server_args.speculative_num_draft_tokens = expected_num_draft_tokens


class DecoupledDraftSpecAlgo(CustomSpecAlgo):
    def is_decoupled_draft(self) -> bool:
        return True

    def supports_spec_v2(self) -> bool:
        return False

    @staticmethod
    def validate_server_args(server_args: ServerArgs) -> None:
        if server_args.enable_dp_attention:
            raise ValueError(
                "decoupled speculative decoding does not support dp attention."
            )
        if server_args.dp_size != 1:
            raise ValueError("decoupled drafter currently requires dp_size == 1.")
        if not server_args.decoupled_spec_bind_endpoint:
            raise ValueError(
                "decoupled speculative decoding requires --decoupled-spec-bind-endpoint."
            )
        if not server_args.decoupled_spec_connect_endpoints:
            raise ValueError(
                "decoupled speculative decoding requires a non-empty "
                "--decoupled-spec-connect-endpoints list."
            )
        if server_args.decoupled_spec_rank is None:
            raise ValueError(
                "decoupled speculative decoding requires --decoupled-spec-rank."
            )
        if int(server_args.decoupled_spec_rank) < 0:
            raise ValueError("--decoupled-spec-rank must be non-negative.")

        if server_args.max_running_requests is None:
            server_args.max_running_requests = 64
            logger.warning(
                "Max running requests is reset to 64 for decoupled speculative decoding. "
                "You can override this by explicitly setting --max-running-requests."
            )

        if not server_args.disable_overlap_schedule:
            server_args.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled for decoupled speculative decoding."
            )

        if not server_args.disable_radix_cache:
            server_args.disable_radix_cache = True
            logger.warning(
                "Radix cache is disabled for decoupled drafter."
            )

        if server_args.mamba_scheduler_strategy != "no_buffer":
            server_args.mamba_scheduler_strategy = "no_buffer"
            logger.warning(
                "Mamba extra buffer is disabled for decoupled drafter engine. "
                "Falling back to --mamba-scheduler-strategy no_buffer."
            )

        if server_args.enable_mixed_chunk:
            server_args.enable_mixed_chunk = False
            logger.warning(
                "Mixed chunked prefill is disabled for decoupled speculative decoding."
            )

        if server_args.speculative_num_steps is None:
            raise ValueError(
                "decoupled speculative decoding requires speculative_num_steps to be set."
            )

        if (
            server_args.speculative_eagle_topk is not None
            and server_args.speculative_eagle_topk != 1
        ):
            raise ValueError(
                "decoupled speculative decoding currently only supports "
                "speculative_eagle_topk == 1."
            )
        server_args.speculative_eagle_topk = 1

        expected_num_draft_tokens = int(server_args.speculative_num_steps) + 1
        if server_args.speculative_num_draft_tokens != expected_num_draft_tokens:
            logger.warning(
                "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 "
                "for decoupled speculative decoding."
            )
            server_args.speculative_num_draft_tokens = expected_num_draft_tokens

    def create_worker(self, server_args: ServerArgs) -> Type:
        raise ValueError(
            "decoupled_draft uses the normal TP worker instead of create_worker()."
        )


@SpeculativeAlgorithm.register(
    "DECOUPLED_VERIFY",
    validate_server_args=DecoupledVerifySpecAlgo.validate_server_args,
    spec_class=DecoupledVerifySpecAlgo,
)
def _decoupled_verify_worker_factory(server_args: ServerArgs) -> Type:
    from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

    return VerifyWorker


@SpeculativeAlgorithm.register(
    "DECOUPLED_DRAFT",
    validate_server_args=DecoupledDraftSpecAlgo.validate_server_args,
    spec_class=DecoupledDraftSpecAlgo,
)
def _decoupled_draft_worker_factory(server_args: ServerArgs) -> Type:
    raise ValueError(
        "decoupled_draft uses the normal TP worker instead of create_worker()."
    )


class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    EAGLE_DRAFT = auto()
    EAGLE_DRAFT_EXTEND = auto()
    EAGLE_VERIFY = auto()
    FROZEN_KV_MTP_DRAFT = auto()
    FROZEN_KV_MTP_DRAFT_EXTEND = auto()
    FROZEN_KV_MTP_VERIFY = auto()
    DFLASH_DRAFT = auto()
    DFLASH_VERIFY = auto()
    NGRAM_VERIFY = auto()


class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    def is_draft_input(self) -> bool:
        # FIXME: remove this function which is only used for assertion
        # or use another variable name like `draft_input` to substitute `spec_info`
        return self.spec_input_type in {
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.FROZEN_KV_MTP_DRAFT,
            SpecInputType.FROZEN_KV_MTP_DRAFT_EXTEND,
            SpecInputType.DFLASH_DRAFT,
        }

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.FROZEN_KV_MTP_VERIFY,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        pass

    def get_spec_adjusted_global_num_tokens(
        self, forward_batch: ModelWorkerBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in forward_batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in forward_batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob
