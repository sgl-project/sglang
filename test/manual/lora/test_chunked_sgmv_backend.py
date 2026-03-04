import random
import unittest
from enum import Enum
from typing import List, Optional, Tuple

import torch

from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.triton_ops import (
    chunked_sgmv_lora_expand_forward,
    chunked_sgmv_lora_shrink_forward,
)
from sglang.srt.lora.triton_ops.chunked_sgmv_expand import _chunked_lora_expand_kernel
from sglang.srt.lora.triton_ops.chunked_sgmv_shrink import _chunked_lora_shrink_kernel
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.lora_utils import reference_sgmv_expand, reference_sgmv_shrink

CHUNK_SIZE = 16


def reset_kernel_cache():
    _chunked_lora_shrink_kernel._clear_cache()
    _chunked_lora_expand_kernel._clear_cache()


class BatchComposition(Enum):
    UNIFORM = "uniform"
    MIXED = "mixed"
    SKEWED = "skewed"
    NONE = "_NO_LORA_"


class BatchMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    TARGET_VERIFY = "verify"


class TestChunkedSGMV(unittest.TestCase):

    # Test configuration constants
    RTOL = 1e-3
    ATOL = 1e-3
    DEFAULT_BATCH_SIZE = 8

    def _compare_shrink_outputs(
        self,
        chunked_output: torch.Tensor,
        reference_output: torch.Tensor,
        seq_lengths: List[int],
        lora_assignments: List[int],
        batch_info: LoRABatchInfo,
        num_slices: int,
        test_name: str,
    ):
        """
        Compare only the valid portions of shrink outputs.

        The chunked SGMV shrink kernel only guarantees correctness for
        output[seq_start:seq_end, :rank * num_slices] for each sequence.
        """
        lora_ranks = batch_info.lora_ranks.cpu().numpy()

        token_offset = 0
        for seq_idx, (lora_idx, seq_len) in enumerate(
            zip(lora_assignments, seq_lengths)
        ):
            if seq_len == 0:
                continue

            rank = lora_ranks[lora_idx]

            if rank > 0:
                # Only compare the valid columns for this sequence
                valid_cols = num_slices * rank

                chunked_seq = chunked_output[
                    token_offset : token_offset + seq_len, :valid_cols
                ]
                reference_seq = reference_output[
                    token_offset : token_offset + seq_len, :valid_cols
                ]

                torch.testing.assert_close(
                    chunked_seq,
                    reference_seq,
                    rtol=self.RTOL,
                    atol=self.ATOL,
                    msg=f"Shrink operation failed for {test_name}, sequence {seq_idx} ({lora_idx})",
                )

            token_offset += seq_len

    def setUp(self):
        """Set up common test parameters"""
        torch.manual_seed(42)
        random.seed(42)

        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.input_dim = 2560  # Hidden dimension
        self.max_seq_len = 1024

        # LoRA configurations: name -> (rank, output_q, output_k, output_v)
        self.lora_configs = {
            "lora_A": (8, 4096, 1024, 1024),
            "lora_B": (16, 4096, 1024, 1024),
            "lora_C": (32, 4096, 1024, 1024),
            "_NO_LORA_": (0, 4096, 1024, 1024),
        }

        # QKV slice offsets: 4096 (Q) + 1024 (K) + 1024 (V) = 6144 total
        self.slice_offsets = torch.tensor(
            [0, 4096, 5120, 6144], dtype=torch.int32, device=self.device
        )
        self.max_slice_size = 4096

    def generate_sequence_lengths(
        self,
        batch_size: int,
        batch_mode: BatchMode = BatchMode.PREFILL,
        min_len: int = 1,
        max_len: int = None,
    ) -> List[int]:
        """Generate sequence lengths for a batch based on mode"""
        if batch_mode == BatchMode.DECODE:
            return [1] * batch_size
        else:
            if max_len is None:
                max_len = self.max_seq_len
            return [random.randint(min_len, max_len) for _ in range(batch_size)]

    def create_lora_weights(
        self, lora_name: str, include_missing_k: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create LoRA A and B weights for given configuration"""
        rank, out_q, out_k, out_v = self.lora_configs[lora_name]

        if rank == 0:
            lora_a = torch.empty(
                0, self.input_dim, dtype=self.dtype, device=self.device
            )
            lora_b = torch.empty(
                out_q + out_k + out_v, 0, dtype=self.dtype, device=self.device
            )
            return lora_a, lora_b

        # Create LoRA A weights (3 slices for QKV)
        lora_a = torch.randn(
            3 * rank, self.input_dim, dtype=self.dtype, device=self.device
        )

        if include_missing_k:
            lora_a[rank : 2 * rank, :] = 0.0

        # Create LoRA B weights (stacked Q, K, V)
        total_output_dim = out_q + out_k + out_v
        lora_b = torch.randn(
            total_output_dim, rank, dtype=self.dtype, device=self.device
        )

        if include_missing_k:
            lora_b[out_q : out_q + out_k, :] = 0.0

        return lora_a, lora_b

    def create_batch_info(
        self,
        lora_names: List[str],
        seq_lengths: List[int],
        lora_assignments: List[Optional[int]],
        batch_mode: BatchMode = BatchMode.PREFILL,
    ) -> LoRABatchInfo:
        """Create LoRABatchInfo using the same logic as chunked backend"""
        lora_ranks = [self.lora_configs[name][0] for name in lora_names]

        def create_mock_batch():
            # Create a minimal mock ForwardBatch for the test
            class MockForwardBatch:
                def __init__(self, batch_size, seq_lengths, device):
                    self.batch_size = batch_size
                    self.extend_seq_lens = torch.tensor(
                        seq_lengths, dtype=torch.int32, device=device
                    )
                    self.extend_seq_lens_cpu = seq_lengths
                    self.forward_mode = MockForwardMode()

            class MockForwardMode:
                def is_extend(self):
                    return batch_mode == BatchMode.PREFILL

                def is_decode(self):
                    return batch_mode == BatchMode.DECODE

                def is_target_verify(self):
                    return batch_mode == BatchMode.TARGET_VERIFY

                def is_prefill(self):
                    return self.is_extend()

            return MockForwardBatch(len(seq_lengths), seq_lengths, self.device)

        mock_batch = create_mock_batch()

        # Use the same functions as chunked backend
        permutation, weights_reordered = ChunkedSgmvLoRABackend._get_permutation(
            lora_assignments, mock_batch
        )

        # Create a minimal backend instance to access _get_segments_info
        mock_server_args = type(
            "ServerArgs", (object,), {"max_lora_chunk_size": "MOCK_NEVER_USED"}
        )
        mock_backend = ChunkedSgmvLoRABackend(
            max_loras_per_batch=8, device=self.device, server_args=mock_server_args
        )
        weight_indices_list, seg_indptr = mock_backend._get_segments_info(
            weights_reordered,
            chunk_size=CHUNK_SIZE,
        )

        scalings = [1.0] * len(lora_names)
        seg_indptr_tensor = seg_indptr.to(self.device)
        weight_indices_tensor = weight_indices_list.to(self.device)
        lora_ranks_tensor = (
            torch.tensor(lora_ranks, dtype=torch.int32, device=self.device)
            if lora_ranks
            else torch.empty(0, dtype=torch.int32, device=self.device)
        )
        scalings_tensor = (
            torch.tensor(scalings, dtype=torch.float32, device=self.device)
            if scalings
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        permutation_tensor = permutation.to(
            self.device, dtype=torch.int32
        )  # Convert to int32 for LoRABatchInfo
        seq_lens_tensor = torch.tensor(
            seq_lengths, dtype=torch.int32, device=self.device
        )

        return LoRABatchInfo(
            use_cuda_graph=False,
            bs=len(seq_lengths),
            num_segments=len(weight_indices_list),  # Number of segments, not sequences!
            seg_indptr=seg_indptr_tensor,
            weight_indices=weight_indices_tensor,
            lora_ranks=lora_ranks_tensor,
            scalings=scalings_tensor,
            seg_lens=seq_lens_tensor,  # Original sequence lengths for reference
            max_len=CHUNK_SIZE,
            permutation=permutation_tensor,  # Token reordering permutation
        )

    def stack_lora_weights(
        self, weight_list: List[torch.Tensor], is_lora_a: bool
    ) -> torch.Tensor:
        """Stack LoRA weights from different adapters into a single tensor"""
        if not weight_list:
            return torch.empty(0, 0, 0, dtype=self.dtype, device=self.device)

        first_non_empty = next((w for w in weight_list if w.numel() > 0), None)
        if first_non_empty is None:
            return torch.empty(
                len(weight_list), 0, 0, dtype=self.dtype, device=self.device
            )
        if is_lora_a:
            # LoRA A: (slice_num * rank, input_dim) -> (num_loras, slice_num * max_rank, input_dim)
            max_rank = max(w.shape[0] // 3 if w.numel() > 0 else 0 for w in weight_list)
            final_shape = (len(weight_list), 3 * max_rank, self.input_dim)
        else:
            # LoRA B: (output_dim, rank) -> (num_loras, output_dim, max_rank)
            max_rank = max(w.shape[1] if w.numel() > 0 else 0 for w in weight_list)
            output_dim = first_non_empty.shape[0]
            final_shape = (len(weight_list), output_dim, max_rank)

        stacked = torch.zeros(final_shape, dtype=self.dtype, device=self.device)

        for i, weight in enumerate(weight_list):
            if weight.numel() > 0:
                if is_lora_a:
                    stacked[i, : weight.shape[0], :] = weight
                else:
                    stacked[i, :, : weight.shape[1]] = weight

        return stacked

    def create_test_batch(
        self,
        batch_composition: BatchComposition,
        batch_size: int,
        batch_mode: BatchMode = BatchMode.PREFILL,
        include_missing_k: bool = False,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        LoRABatchInfo,
        List[int],
        List[str],
    ]:
        """Create test batch with specified composition and mode"""

        # Reset kernel cache to avoid cross-test contamination
        reset_kernel_cache()

        seq_lengths = self.generate_sequence_lengths(
            batch_size, batch_mode, 1, self.max_seq_len
        )
        if batch_composition == BatchComposition.UNIFORM:
            lora_names = ["lora_A"]
            lora_assignments = [lora_names.index("lora_A")] * batch_size
        elif batch_composition == BatchComposition.MIXED:
            lora_names = ["lora_A", "lora_B", "lora_C", None]
            lora_assignments = [(i % len(lora_names)) for i in range(batch_size)]
        elif batch_composition == BatchComposition.SKEWED:
            lora_names = ["lora_A", "lora_B"]
            num_minority = max(1, batch_size // 8)
            lora_assignments = [lora_names.index("lora_A")] * num_minority + [
                lora_names.index("lora_B")
            ] * (batch_size - num_minority)
            random.shuffle(lora_assignments)
        elif batch_composition == BatchComposition.NONE:
            lora_names = [None]
            lora_assignments = [0] * batch_size
        else:
            raise ValueError(f"Unknown batch composition: {batch_composition}")

        total_seq_len = sum(seq_lengths)
        x = torch.randn(
            total_seq_len, self.input_dim, dtype=self.dtype, device=self.device
        )

        normalized_lora_names = [
            "_NO_LORA_" if name is None else name for name in lora_names
        ]
        weights = []
        for lora_name in normalized_lora_names:
            weights.append(self.create_lora_weights(lora_name, include_missing_k))

        batch_info = self.create_batch_info(
            normalized_lora_names, seq_lengths, lora_assignments, batch_mode
        )

        return x, weights, batch_info, seq_lengths, lora_assignments

    def run_test_comparison(
        self,
        x: torch.Tensor,
        weights: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_info: LoRABatchInfo,
        seq_lengths: List[int],
        lora_assignments: List[int],
        test_name: str,
    ):
        """Run comparison between chunked and reference implementations"""
        if not weights:  # Handle case with no LoRA weights
            return

        lora_assignments_tensor = torch.tensor(
            lora_assignments, dtype=torch.int32, device="cpu"
        )
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = batch_info.lora_ranks.detach().cpu()
        scalings_tensor = batch_info.scalings.detach().cpu()

        # Stack LoRA A weights
        lora_a_weights = [weight[0] for weight in weights]
        stacked_lora_a = self.stack_lora_weights(lora_a_weights, is_lora_a=True)

        # Stack LoRA B weights
        lora_b_weights = [weight[1] for weight in weights]
        stacked_lora_b = self.stack_lora_weights(lora_b_weights, is_lora_a=False)

        # Test shrink operation
        chunked_shrink = chunked_sgmv_lora_shrink_forward(
            x, stacked_lora_a, batch_info, num_slices=3
        )
        reference_shrink = reference_sgmv_shrink(
            x,
            stacked_lora_a,
            lora_assignments_tensor,
            seq_lengths_tensor,
            lora_ranks_tensor,
            scalings_tensor,
            num_slices=3,
        )

        # Only compare valid portions of shrink output (first rank * num_slices columns per sequence)
        self._compare_shrink_outputs(
            chunked_shrink,
            reference_shrink,
            seq_lengths,
            lora_assignments,
            batch_info,
            num_slices=3,
            test_name=test_name,
        )

        # Test expand operation
        chunked_expand = chunked_sgmv_lora_expand_forward(
            reference_shrink,
            stacked_lora_b,
            batch_info,
            self.slice_offsets,
            self.max_slice_size,
            base_output=None,
        )
        reference_expand = reference_sgmv_expand(
            reference_shrink,
            stacked_lora_b,
            lora_assignments_tensor,
            seq_lengths_tensor,
            lora_ranks_tensor,
            self.slice_offsets,
        )

        torch.testing.assert_close(
            chunked_expand,
            reference_expand,
            rtol=self.RTOL,
            atol=self.ATOL,
            msg=f"Expand operation failed for {test_name}",
        )

    # === Basic Operations Tests ===

    def test_shrink_basic(self):
        """Test basic shrink operation against PyTorch reference"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(BatchComposition.UNIFORM, batch_size)
                )

                lora_assignments_tensor = torch.tensor(
                    lora_assignments, dtype=torch.int32, device="cpu"
                )
                seq_lengths_tensor = torch.tensor(
                    seq_lengths, dtype=torch.int32, device="cpu"
                )
                lora_ranks_tensor = batch_info.lora_ranks.detach().cpu()
                scalings_tensor = batch_info.scalings.detach().cpu()

                lora_a_weights = [weight[0] for weight in weights]
                stacked_lora_a = self.stack_lora_weights(lora_a_weights, is_lora_a=True)

                chunked_shrink = chunked_sgmv_lora_shrink_forward(
                    x, stacked_lora_a, batch_info, num_slices=3
                )
                reference_shrink = reference_sgmv_shrink(
                    x,
                    stacked_lora_a,
                    lora_assignments_tensor,
                    seq_lengths_tensor,
                    lora_ranks_tensor,
                    scalings_tensor,
                    num_slices=3,
                )

                torch.testing.assert_close(
                    chunked_shrink, reference_shrink, rtol=self.RTOL, atol=self.ATOL
                )

    def test_expand_basic(self):
        """Test basic expand operation against PyTorch reference"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(BatchComposition.UNIFORM, batch_size)
                )

                lora_assignments_tensor = torch.tensor(
                    lora_assignments, dtype=torch.int32, device="cpu"
                )
                seq_lengths_tensor = torch.tensor(
                    seq_lengths, dtype=torch.int32, device="cpu"
                )
                lora_ranks_tensor = batch_info.lora_ranks.detach().cpu()
                scalings_tensor = batch_info.scalings.detach().cpu()

                lora_a_weights = [weight[0] for weight in weights]
                stacked_lora_a = self.stack_lora_weights(lora_a_weights, is_lora_a=True)

                intermediate = reference_sgmv_shrink(
                    x,
                    stacked_lora_a,
                    lora_assignments_tensor,
                    seq_lengths_tensor,
                    lora_ranks_tensor,
                    scalings_tensor,
                    num_slices=3,
                )

                lora_b_weights = [weight[1] for weight in weights]
                stacked_lora_b = self.stack_lora_weights(
                    lora_b_weights, is_lora_a=False
                )

                chunked_expand = chunked_sgmv_lora_expand_forward(
                    intermediate,
                    stacked_lora_b,
                    batch_info,
                    self.slice_offsets,
                    self.max_slice_size,
                    base_output=None,
                )
                reference_expand = reference_sgmv_expand(
                    intermediate,
                    stacked_lora_b,
                    lora_assignments_tensor,
                    seq_lengths_tensor,
                    lora_ranks_tensor,
                    self.slice_offsets,
                )

                torch.testing.assert_close(
                    chunked_expand, reference_expand, rtol=self.RTOL, atol=self.ATOL
                )

    # === QKV Operations Test ===

    def test_qkv_missing_projections(self):
        """Test QKV operations with missing k_proj (Qwen3 scenario)"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(
                        BatchComposition.MIXED, batch_size, include_missing_k=True
                    )
                )
                self.run_test_comparison(
                    x,
                    weights,
                    batch_info,
                    seq_lengths,
                    lora_assignments,
                    f"QKV missing k_proj batch_size={batch_size}",
                )

    # === Batch Composition Tests ===

    def test_uniform_lora_batch(self):
        """All sequences use same LoRA, random sequence lengths"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(BatchComposition.UNIFORM, batch_size)
                )
                self.run_test_comparison(
                    x,
                    weights,
                    batch_info,
                    seq_lengths,
                    lora_assignments,
                    f"uniform batch_size={batch_size}",
                )

    def test_evenly_mixed_lora_batch(self):
        """Sequences evenly distributed across LoRAs, random lengths"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(BatchComposition.MIXED, batch_size)
                )
                self.run_test_comparison(
                    x,
                    weights,
                    batch_info,
                    seq_lengths,
                    lora_assignments,
                    f"mixed batch_size={batch_size}",
                )

    def test_highly_skewed_lora_batch(self):
        """Highly uneven LoRA distribution, random lengths"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(BatchComposition.SKEWED, batch_size)
                )
                self.run_test_comparison(
                    x,
                    weights,
                    batch_info,
                    seq_lengths,
                    lora_assignments,
                    f"skewed batch_size={batch_size}",
                )

    # === Decode Mode Tests ===

    def test_decode_uniform_lora_batch(self):
        """Decode mode: All sequences use same LoRA, all length 1"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(
                        BatchComposition.UNIFORM, batch_size, BatchMode.DECODE
                    )
                )
                self.run_test_comparison(
                    x,
                    weights,
                    batch_info,
                    seq_lengths,
                    lora_assignments,
                    f"decode uniform batch_size={batch_size}",
                )

    def test_decode_mixed_lora_batch(self):
        """Decode mode: Sequences distributed across LoRAs, all length 1"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(
                        BatchComposition.MIXED, batch_size, BatchMode.DECODE
                    )
                )
                self.run_test_comparison(
                    x,
                    weights,
                    batch_info,
                    seq_lengths,
                    lora_assignments,
                    f"decode mixed batch_size={batch_size}",
                )

    def test_decode_skewed_lora_batch(self):
        """Decode mode: Highly uneven LoRA distribution, all length 1"""
        for batch_size in [1, 2, 16, 64]:
            with self.subTest(batch_size=batch_size):
                x, weights, batch_info, seq_lengths, lora_assignments = (
                    self.create_test_batch(
                        BatchComposition.SKEWED, batch_size, BatchMode.DECODE
                    )
                )
                self.run_test_comparison(
                    x,
                    weights,
                    batch_info,
                    seq_lengths,
                    lora_assignments,
                    f"decode skewed batch_size={batch_size}",
                )


if __name__ == "__main__":
    unittest.main()
