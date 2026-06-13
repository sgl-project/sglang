from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

import torch

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
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker


logger = logging.getLogger(__name__)


def _handle_dflash(server_args: "ServerArgs") -> None:
    if server_args.enable_dp_attention:
        raise ValueError(
            "Currently DFLASH speculative decoding does not support dp attention."
        )

    if server_args.pp_size != 1:
        raise ValueError(
            "Currently DFLASH speculative decoding only supports pp_size == 1."
        )

    if server_args.speculative_draft_model_path is None:
        raise ValueError(
            "DFLASH speculative decoding requires setting --speculative-draft-model-path."
        )

    # DFLASH does not use EAGLE-style `num_steps`/`topk`, but those fields still
    # affect generic scheduler/KV-cache accounting (buffer sizing, KV freeing,
    # RoPE reservation). Force them to 1 to avoid surprising memory behavior.
    #
    # For DFlash, the natural unit is `block_size` (verify window length).
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = 1
    elif int(server_args.speculative_num_steps) != 1:
        logger.warning(
            "DFLASH only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            server_args.speculative_num_steps,
        )
        server_args.speculative_num_steps = 1

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif int(server_args.speculative_eagle_topk) != 1:
        logger.warning(
            "DFLASH only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            server_args.speculative_eagle_topk,
        )
        server_args.speculative_eagle_topk = 1

    if server_args.speculative_dflash_block_size is not None:
        if int(server_args.speculative_dflash_block_size) <= 0:
            raise ValueError(
                "DFLASH requires --speculative-dflash-block-size to be positive, "
                f"got {server_args.speculative_dflash_block_size}."
            )
        if server_args.speculative_num_draft_tokens is not None and int(
            server_args.speculative_num_draft_tokens
        ) != int(server_args.speculative_dflash_block_size):
            raise ValueError(
                "Both --speculative-num-draft-tokens and --speculative-dflash-block-size are set "
                "but they differ. For DFLASH they must match. "
                f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}, "
                f"speculative_dflash_block_size={server_args.speculative_dflash_block_size}."
            )
        server_args.speculative_num_draft_tokens = int(
            server_args.speculative_dflash_block_size
        )

    if server_args.speculative_num_draft_tokens is None:
        from sglang.srt.speculative.dflash_utils import (
            parse_dflash_draft_config,
        )

        model_override_args = json.loads(server_args.json_model_override_args)
        inferred_block_size = None
        try:
            from sglang.srt.utils.hf_transformers_utils import get_config

            draft_hf_config = get_config(
                server_args.speculative_draft_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.speculative_draft_model_revision,
                model_override_args=model_override_args,
            )
            inferred_block_size = parse_dflash_draft_config(
                draft_hf_config=draft_hf_config
            ).resolve_block_size(default=None)
        except Exception as e:
            logger.warning(
                "Failed to infer DFLASH block_size from draft model config; "
                "defaulting speculative_num_draft_tokens to 16. Error: %s",
                e,
            )

        if inferred_block_size is None:
            inferred_block_size = 16
            logger.warning(
                "speculative_num_draft_tokens is not set; defaulting to %d for DFLASH.",
                inferred_block_size,
            )
        server_args.speculative_num_draft_tokens = inferred_block_size

    if server_args.speculative_draft_window_size is not None:
        draft_tokens = int(server_args.speculative_num_draft_tokens)
        if server_args.speculative_draft_window_size < draft_tokens:
            raise ValueError(
                "--speculative-draft-window-size must be >= "
                "--speculative-num-draft-tokens (block_size). "
                f"window_size={server_args.speculative_draft_window_size}, block_size={draft_tokens}."
            )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using dflash speculative decoding."
        )


def _handle_frozen_kv_mtp(server_args: "ServerArgs") -> None:
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "Frozen-KV MTP speculative decoding."
        )


def _handle_eagle_family(server_args: "ServerArgs") -> None:
    if (
        server_args.speculative_algorithm == "STANDALONE"
        and server_args.enable_dp_attention
    ):
        # TODO: support dp attention for standalone speculative decoding
        raise ValueError(
            "Currently standalone speculative decoding does not support dp attention."
        )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.disable_overlap_schedule:
        logger.warning(
            "Non-overlap (synchronous) spec v2 is used for eagle/eagle3/standalone "
            "speculative decoding."
        )
    else:
        logger.warning(
            "Overlap spec v2 is enabled by default for eagle/eagle3/standalone speculative decoding."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "eagle speculative decoding."
        )

    model_arch = server_args.get_model_config().hf_config.architectures[0]
    if model_arch in [
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "BailingMoeV2_5ForCausalLM",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "HYV3ForCausalLM",
    ]:
        if server_args.speculative_draft_model_path is None:
            server_args.speculative_draft_model_path = server_args.model_path
            server_args.speculative_draft_model_revision = server_args.revision
        else:
            if model_arch not in [
                "MistralLarge3ForCausalLM",
                "PixtralForConditionalGeneration",
            ]:
                logger.warning(
                    "DeepSeek MTP does not require setting speculative_draft_model_path."
                )

    if (
        not server_args.speculative_adaptive
        and server_args.speculative_num_steps is None
    ):
        assert (
            server_args.speculative_eagle_topk is None
            and server_args.speculative_num_draft_tokens is None
        )

        (
            server_args.speculative_num_steps,
            server_args.speculative_eagle_topk,
            server_args.speculative_num_draft_tokens,
        ) = _auto_choose_speculative_params(server_args, model_arch)

    if (
        server_args.attention_backend == "trtllm_mha"
        or server_args.decode_attention_backend == "trtllm_mha"
        or server_args.prefill_attention_backend == "trtllm_mha"
    ):
        if server_args.speculative_eagle_topk > 1:
            raise ValueError(
                "trtllm_mha backend only supports topk = 1 for speculative decoding."
            )

    if (
        server_args.speculative_eagle_topk == 1
        and server_args.speculative_num_draft_tokens
        != server_args.speculative_num_steps + 1
    ):
        logger.warning(
            "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
        )
        server_args.speculative_num_draft_tokens = server_args.speculative_num_steps + 1

    # topk > 1 + page_size > 1 needs the two-pass cascade draft-decode (shared prefix
    # pass + per-branch expand pass with prefix-tail dup). Only these backends implement
    # it; flashmla / trtllm_mla / cutlass_mla can't express the per-branch tree, so reject.
    _PAGE_TREE_SPEC_BACKENDS = ("flashinfer", "fa3", "triton")
    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend not in _PAGE_TREE_SPEC_BACKENDS
    ):
        raise ValueError(
            f"speculative_eagle_topk > 1 with page_size > 1 is only supported on "
            f"{_PAGE_TREE_SPEC_BACKENDS}; got attention_backend="
            f"{server_args.attention_backend!r}. Use page_size == 1 or one of those backends."
        )


def _handle_ngram(server_args: "ServerArgs") -> None:
    if not server_args.device.startswith("cuda"):
        raise ValueError("Ngram speculative decoding only supports CUDA device.")

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    server_args.enable_mixed_chunk = False
    server_args.speculative_eagle_topk = server_args.speculative_ngram_max_bfs_breadth
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = 12
        logger.warning(
            "speculative_num_draft_tokens is set to 12 by default for ngram speculative decoding. "
            "You can override this by explicitly setting --speculative-num-draft-tokens."
        )
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = (
            server_args.speculative_num_draft_tokens
            // server_args.speculative_eagle_topk
        )
    if server_args.speculative_ngram_external_corpus_path is not None:
        if server_args.speculative_ngram_external_sam_budget <= 0:
            raise ValueError(
                "--speculative-ngram-external-sam-budget must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if server_args.speculative_ngram_external_corpus_max_tokens <= 0:
            raise ValueError(
                "--speculative-ngram-external-corpus-max-tokens must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if (
            server_args.speculative_ngram_external_sam_budget
            > server_args.speculative_num_draft_tokens - 1
        ):
            raise ValueError(
                "speculative_ngram_external_sam_budget must be less than or equal to "
                f"speculative_num_draft_tokens - 1 ({server_args.speculative_num_draft_tokens - 1})."
            )
    logger.warning(
        "The mixed chunked prefill are disabled because of "
        "using ngram speculative decoding."
    )

    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend != "flashinfer"
    ):
        raise ValueError(
            f"speculative_eagle_topk({server_args.speculative_eagle_topk}) > 1 "
            f"with page_size({server_args.page_size}) > 1 is unstable "
            "and produces incorrect results for paged attention backends. "
            "This combination is only supported for the 'flashinfer' backend."
        )
    if server_args.enable_dp_attention:
        # TODO: support dp attention for ngram speculative decoding
        raise ValueError(
            "Currently ngram speculative decoding does not support dp attention."
        )


def _auto_choose_speculative_params(
    server_args: "ServerArgs", model_arch: str
) -> tuple:
    """
    Automatically choose the parameters for speculative decoding.

    You can tune them on your own models and prompts with scripts/playground/bench_speculative.py
    """
    if server_args.speculative_algorithm == "STANDALONE":
        return (3, 1, 4)
    if model_arch in ["LlamaForCausalLM"]:
        return (5, 4, 8)
    elif model_arch in [
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV2ForCausalLM",
        "GptOssForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "BailingMoeV2_5ForCausalLM",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "MiMoV2ForCausalLM",
        "MiMoV2FlashForCausalLM",
    ]:
        return (3, 1, 4)
    elif model_arch in ["Grok1ForCausalLM", "Grok1VForCausalLM"]:
        return (5, 4, 8)
    else:
        return (3, 1, 4)


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

    def is_some(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

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

    def supports_target_verify_for_draft(self) -> bool:
        return self.is_dflash()

    def has_draft_kv(self) -> bool:
        """Whether the draft phase writes KV chains. NGRAM does not (its tree
        lives only in the verify mask), so per-decode KV sizing needs no
        per-topk page rounding; see get_alloc_len_per_decode."""
        return not self.is_ngram()

    def carries_draft_hidden_states(self) -> bool:
        """Whether the disagg prefill->decode transfer carries draft hidden
        states (EAGLE-family only; STANDALONE's vanilla draft ignores them)."""
        return self.is_eagle()

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
    ) -> FutureMap:
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        server_args: ServerArgs,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        if self.is_eagle():
            from sglang.srt.speculative.eagle_disaggregation import (
                build_eagle_disagg_draft_input,
            )

            return build_eagle_disagg_draft_input(
                batch, server_args, last_tokens_tensor, future_map
            )
        return None

    def need_topk(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def handle_server_args(self, server_args: "ServerArgs") -> None:
        """Hook for per-algorithm server args mutation.

        In-place updated.
        """
        if self.is_dflash():
            _handle_dflash(server_args)
        elif self.is_frozen_kv_mtp():
            _handle_frozen_kv_mtp(server_args)
        elif self.is_eagle() or self.is_standalone():
            _handle_eagle_family(server_args)
        elif self.is_ngram():
            _handle_ngram(server_args)

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        return num_draft_tokens

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        if self.is_dflash():
            # V2 worker drives both overlap and non-overlap (scheduler runs it
            # synchronously when overlap is disabled), same as EAGLE.
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            return DFlashWorkerV2

        if self.is_frozen_kv_mtp():
            # V2 worker drives both overlap and non-overlap (scheduler runs it
            # synchronously when overlap is disabled), same as EAGLE.
            from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
                FrozenKVMTPWorkerV2,
            )

            return FrozenKVMTPWorkerV2

        # EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER always use the V2 worker,
        # even with overlap disabled (scheduler drives it synchronously).
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2

        elif self.is_eagle():
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            return EAGLEWorkerV2
        elif self.is_standalone():
            from sglang.srt.speculative.standalone_worker_v2 import (
                StandaloneWorkerV2,
            )

            return StandaloneWorkerV2
        elif self.is_ngram():
            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


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

    # Cross-algorithm phase guards. Used by attention backends and
    # ForwardBatch padding logic to dispatch on phase without hardcoding the
    # specific algo class (EAGLE / FROZEN_KV_MTP / DFLASH / NGRAM each have
    # their own draft / verify SpecInput subclasses).
    def is_draft_input(self) -> bool:
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
        self, batch: ScheduleBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob


def create_dummy_verify_input(
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    custom_mask: torch.Tensor,
    num_tokens_per_bs: int,
    is_draft_worker: bool,
) -> Optional[SpecInput]:
    """Dummy verify ``SpecInput`` for CUDA-graph capture (per-algorithm dispatch)."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    spec_info = None
    if spec_algorithm.is_eagle() or spec_algorithm.is_standalone():
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        if is_draft_worker:
            raise RuntimeError("This should not happen.")
        else:
            spec_info = EagleVerifyInput(
                draft_token=None,
                custom_mask=custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_cum_len=None,
                spec_steps=server_args.speculative_num_steps,
                topk=server_args.speculative_eagle_topk,
                draft_token_num=server_args.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
    elif spec_algorithm.is_dflash():
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
        spec_info = DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=server_args.speculative_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=(
                CaptureHiddenMode.NULL if is_draft_worker else CaptureHiddenMode.FULL
            ),
        )

    elif spec_algorithm.is_ngram():
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        spec_info = NgramVerifyInput(
            draft_token=None,
            custom_mask=custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            draft_token_num=num_tokens_per_bs,
        )
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

    return spec_info
