import logging
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


class StandaloneWorker(EAGLEWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # TODO: Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with (
            empty_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Validate that the draft model shares the same vocabulary as the target model.
        # vocab_size equality is a quick necessary check, but not sufficient: two models
        # can have the same vocab_size with different token strings.  We also compare the
        # full token-to-id mapping of both tokenizers (the same approach used by
        # HuggingFace Transformers to detect homogeneous vs heterogeneous vocabularies).
        self._validate_vocab_compatibility(
            target_vocab_size=target_worker.model_runner.model_config.vocab_size,
            target_tokenizer=target_worker.tokenizer,
        )

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def _validate_vocab_compatibility(
        self,
        target_vocab_size: int,
        target_tokenizer,
    ) -> None:
        """Raise ValueError if the draft and target vocabularies are incompatible.

        STANDALONE requires both models to share the same vocabulary.  Two
        conditions are checked:
        1. ``vocab_size`` must be identical (fast check).
        2. The full token-to-id mapping from ``get_vocab()`` must be identical
           (catches same-size but different-content tokenizers).

        Either tokenizer being ``None`` (e.g. ``--skip-tokenizer-init``) or a
        tokenizer type that does not implement ``get_vocab()`` (e.g.
        ``TiktokenTokenizer``) skips the deep mapping check.

        The ``get_vocab()`` comparison is O(vocab_size) but runs only once at
        startup; on a 256 k-token vocabulary it takes ~300 ms, which is
        negligible relative to model-loading time.
        """
        draft_vocab_size = self.draft_model_runner.model_config.vocab_size
        draft_tokenizer = self.tokenizer
        if target_vocab_size != draft_vocab_size:
            raise ValueError(
                f"STANDALONE speculative decoding requires the draft model to share the "
                f"same vocabulary as the target model, but got "
                f"target vocab_size={target_vocab_size} and "
                f"draft vocab_size={draft_vocab_size}. "
                f"Use a draft model with a matching vocabulary, or a speculative "
                f"algorithm that supports heterogeneous vocabularies."
            )
        if (
            target_tokenizer is not None
            and draft_tokenizer is not None
            and hasattr(target_tokenizer, "get_vocab")
            and hasattr(draft_tokenizer, "get_vocab")
            and target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
        ):
            raise ValueError(
                "STANDALONE speculative decoding requires the draft model to share the "
                "same vocabulary as the target model, but the two tokenizers have "
                "different token-to-id mappings even though their vocab sizes match. "
                "Use a draft model with a matching vocabulary, or a speculative "
                "algorithm that supports heterogeneous vocabularies."
            )
