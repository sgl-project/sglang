from typing import Tuple, Union

import torch

from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseLoRABackend:
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.

    Args:
        max_loras_per_batch: maximum number of different lora weights
                             that can be applied in a single forward batch.
        device: the device where the backend runs.
    """

    def __init__(self, max_loras_per_batch: int, device: torch.device):
        self.max_loras_per_batch = max_loras_per_batch
        self.device = device
        # lm_head receives pruned hidden states whose shape differs from the
        # original extend_seq_lens used to build the main ``batch_info``.
        # Backends that support lm_head LoRA should populate this field in
        # ``prepare_lora_batch`` so that ``ParallelLMHeadWithLoRA`` can swap
        # it in during its forward pass.
        self.lm_head_batch_info = None
        # Saved full-pruned version so the chunked-logprobs path can restore
        # it after iterating over individual chunks.
        self._lm_head_batch_info_full = None

    def run_lora_a_embedding(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        vocab_size: int,
        extra_embeddings: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run LoRA A embedding lookup with CUDA graph support.

        Args:
            input_ids: token IDs with shape (s,), where s is the sum of all sequence lengths
            weights: LoRA A embedding weights with shape (num_loras, rank, vocab_size)
            vocab_size: base vocabulary size (tokens >= vocab_size are extra tokens)
            extra_embeddings: extra token embeddings with shape (num_loras, num_extra_tokens, rank)
            Only needed if there are added tokens beyond base vocabulary.

        Returns:
            result with shape (s, rank)
        """
        pass

    def run_extra_token_embedding(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor,
        extra_embeddings: torch.Tensor,
        vocab_size: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply extra token embeddings to output in-place.

        Args:
            input_ids: (s,) token IDs
            output: (s, embed_dim) output tensor to be modified
            extra_embeddings: (num_loras, num_extra_tokens, embed_dim) extra embeddings
            vocab_size: base vocabulary size

        Returns:
            output: modified output tensor
        """
        raise NotImplementedError

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora a modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, c * r, input_dim),
                      here r is lora rank, c is a multiplier for stacked modules (e.g., c=3 for qkv_proj, c=2 for gate_up_proj)
                      usually input_dim is much larger than r
        Returns:
             result with shape (s, c * r)
        """
        pass

    def run_lora_b_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora b modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, r), here s is the sum of all sequence lengths, r is lora rank
             weights: a set of lora weights with shape (num_lora, output_dim, r)
                      usually output_dim is much larger than r
        Returns:
             result with shape (s, output_dim)
        """
        pass

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            qkv_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora,output_dim_q + 2 * output_dim_kv, r)
                        If passed in as a tuple of two tensors, it should contain:
                           a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
                           and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        pass

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for gate_up_proj, usually attached to MergedColumnParallelLayer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            gate_up_lora_a: lora_a module for gate_up_proj, with shape (num_lora, 2 * r, input_dim)
            gate_up_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora, 2 * output_dim, r)
                        If passed in as a tuple, it should contain two tensors with shape (num_lora, output_dim, r)
        Returns:
            result with shape (s, 2 * output_dim)
        """
        pass

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        """Initialize the batch info for CUDA Graph mode.

        This method provides a hook for each backend to conduct its own initialization
        logic for CUDA Graph mode.

        Args:
            cuda_graph_batch_info: the LoRABatchInfo object created in LoraManager
            max_bs_in_cuda_graph: maximum batch size for CUDA Graph mode
            num_tokens_per_bs: number of tokens per sequence (1 for decoding, >1 for target_verify)
        """
        pass

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
    ):
        """Prepare the lora weights and batch info for current forward batch.

        This method provides a hook for each backend to conduct its own preparation
        logic for each forward batch.

        Args:
            forward_batch: the ForwardBatch object for current forward pass
            weight_indices: list of indices of lora weights to be applied for current batch
            lora_ranks: list of lora ranks corresponding to weight_indices
            scalings: list of scaling factors corresponding to weight_indices
            use_cuda_graph: whether to use CUDA Graph for this batch
        """
        pass

    # ------------------------------------------------------------------
    # lm_head pruned batch_info helpers
    # ------------------------------------------------------------------
    # LogitsProcessor prunes hidden states before lm_head, so the default
    # batch_info no longer matches.  ``_compute_lm_head_batch_info``
    # (shared) computes the pruned segmentation;
    # ``_make_lm_head_batch_info`` (overridable) creates the concrete
    # batch-info object for the backend.

    def _compute_lm_head_batch_info(
        self, forward_batch: ForwardBatch, weight_indices: list[int]
    ):
        """Pre-compute ``lm_head_batch_info`` that matches the pruned hidden
        states that ``LogitsProcessor`` will pass to the lm_head layer.

        Three scenarios:
        * **Decode** – no pruning; lm_head sees the same tokens → ``None``.
        * **Extend without logprob** – only last token per seq → seg_lens=[1,…].
        * **Extend with logprob** – pruned by ``extend_logprob_start_lens`` →
          seg_lens = per-sequence pruned lengths.
        """
        if not forward_batch.forward_mode.is_extend():
            self.lm_head_batch_info = None
            self._lm_head_batch_info_full = None
            return

        bs = forward_batch.batch_size

        if (
            forward_batch.return_logprob
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_logprob_start_lens_cpu is not None
        ):
            # Mirrors ``_get_pruned_states()`` in logits_processor.py.
            pruned_seg_lens_list = [
                max(1, ext - start)  # at least 1 for sampling
                for ext, start in zip(
                    forward_batch.extend_seq_lens_cpu,
                    forward_batch.extend_logprob_start_lens_cpu,
                )
            ]
        else:
            # Extend without logprob: only last token per sequence.
            pruned_seg_lens_list = [1] * bs

        self.lm_head_batch_info = self._make_lm_head_batch_info(
            pruned_seg_lens_list, weight_indices
        )
        self._lm_head_batch_info_full = self.lm_head_batch_info

    def _make_lm_head_batch_info(
        self,
        seg_lens_list: list[int],
        weight_indices_list: list[int],
    ) -> LoRABatchInfo:
        """Create a ``LoRABatchInfo`` from *pruned* segment lengths.

        The default implementation builds a standard ``LoRABatchInfo`` suitable
        for the Triton backend.  Backends that require a different type (e.g.
        ``TorchNativeLoRABatchInfo``) should override this method.
        """
        num_segs = len(seg_lens_list)
        seg_lens = torch.tensor(seg_lens_list, dtype=torch.int32, device=self.device)
        seg_indptr = torch.zeros(num_segs + 1, dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

        wi_tensor = torch.tensor(
            weight_indices_list[:num_segs], dtype=torch.int32, device=self.device
        )

        return LoRABatchInfo(
            bs=num_segs,
            num_segments=num_segs,
            max_len=max(seg_lens_list) if seg_lens_list else 0,
            use_cuda_graph=False,
            seg_lens=seg_lens,
            seg_indptr=seg_indptr,
            weight_indices=wi_tensor,
            lora_ranks=self.batch_info.lora_ranks,   # shared reference
            scalings=self.batch_info.scalings,        # shared reference
            permutation=None,
        )
