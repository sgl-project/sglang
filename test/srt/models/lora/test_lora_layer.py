# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import multiprocessing as mp
import unittest
from typing import List, Optional

import torch
import torch.nn.functional as F
import random
import numpy as np

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.layers import VocabParallelEmbeddingWithLoRA
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.test_utils import CustomTestCase

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}

DEFAULT_VOCAB_PADDING_SIZE = 64

class TestLoRALayer(CustomTestCase):

    def setUp(self):
        """Set up test parameters and common configurations."""
        seed=1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.max_loras_per_batch = 4
        self.lora_rank = 8
        self.scaling_factor = 1.0 / self.lora_rank
        self.tolerance = 1e-3
        
        self.vocab_sizes = [512, 32000]
        self.embed_dims = [128, 256]
        self.devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        self.num_loras_list = [1, 4]
        
        self.batch_sizes = [1, 4]
        self.seq_lens = [5, 10, 100, 1000]
        self.num_added_tokens = 16

    def _create_base_embedding_layer(self, vocab_size: int, embed_dim: int, device: str) -> VocabParallelEmbedding:
        """Create a base VocabParallelEmbedding layer for testing."""
        base_layer = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            enable_tp=False,
            org_num_embeddings=vocab_size,
        )
        base_layer = base_layer.to(device)
        embedding_data = torch.rand_like(base_layer.weight.data)
        base_layer.weight.data = embedding_data
        return base_layer

    def _create_lora_embedding_layer(self, base_layer: VocabParallelEmbedding, device: str) -> VocabParallelEmbeddingWithLoRA:
        """Create a LoRA embedding layer wrapping the base layer."""
        lora_backend = TritonLoRABackend("triton")
        lora_layer = VocabParallelEmbeddingWithLoRA(base_layer, lora_backend)
        return lora_layer

    def _create_lora_weights(
        self, 
        new_embeddings: torch.Tensor,
        num_loras: int, 
        vocab_size: int, 
        embed_dim: int, 
        num_added_tokens: int,
        device: str
    ) -> tuple:
        """Create LoRA weight matrices for testing."""

        new_embeddings_buffer = new_embeddings.to(device)

        embedding_A_buffer = torch.randn(
            num_loras, self.lora_rank, vocab_size + num_added_tokens,
            dtype=torch.float32, device=device
        ) * 0.1
        
        embedding_B_buffer = torch.randn(
            num_loras, embed_dim, self.lora_rank,
            dtype=torch.float32, device=device
        ) * 0.1
        
        return new_embeddings_buffer, embedding_A_buffer, embedding_B_buffer

    def _create_batch_info(
        self, 
        batch_size: int, 
        seq_lens: List[int], 
        lora_ids: List[Optional[int]], 
        device: str
    ) -> LoRABatchInfo:
        """Create LoRABatchInfo for the test batch."""
        seg_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        
        seg_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        
        weight_indices = torch.tensor(
            [lora_id if lora_id is not None else 0 for lora_id in lora_ids], 
            dtype=torch.int32, device=device
        )
        
        lora_ranks = torch.full(
            (self.max_loras_per_batch,), self.lora_rank, 
            dtype=torch.int32, device=device
        )
        
        scalings = torch.full(
            (self.max_loras_per_batch,), self.scaling_factor, 
            dtype=torch.float32, device=device
        )
        
        max_len = max(seq_lens)
        
        return LoRABatchInfo(
            bs=batch_size,
            seg_lens=seg_lens,
            seg_indptr=seg_indptr,
            max_len=max_len,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
        )

    def _create_test_input(
        self, 
        batch_size: int, 
        seq_lens: List[int], 
        vocab_size: int, 
        num_added_tokens: int,
        device: str
    ) -> torch.Tensor:
        """Create test input with mix of base vocab and added tokens."""
        total_tokens = sum(seq_lens)
        
        input_ids = []
        
        for seq_len in seq_lens:
            base_tokens = torch.randint(0, vocab_size, (seq_len // 2,), device=device)
            added_tokens = torch.randint(vocab_size, vocab_size + num_added_tokens, (seq_len - seq_len // 2,), device=device)
            seq_tokens = torch.cat([base_tokens, added_tokens])
            
            perm = torch.randperm(seq_len, device=device)
            seq_tokens = seq_tokens[perm]
            input_ids.append(seq_tokens)
        
        return torch.cat(input_ids, dim=0)

    def _compute_expected_output(
        self,
        input_ids: torch.Tensor,
        base_layer: VocabParallelEmbedding,
        new_embeddings: torch.Tensor,
        embedding_A_buffer: torch.Tensor,
        embedding_B_buffer: torch.Tensor,
        batch_info: LoRABatchInfo,
        vocab_size: int,
        num_added_tokens: int,
    ) -> torch.Tensor:
        """Manually compute the expected LoRA embedding output using VocabParallelEmbedding + manual LoRA."""
        expected_results: list[torch.Tensor] = []
        org_vocab_size_padded = base_layer.num_embeddings_padded
        
        current_pos = 0
        for i in range(batch_info.bs):
            seq_len = batch_info.seg_lens[i]
            lora_id = batch_info.weight_indices[i].item()
            seq_input = input_ids[current_pos:current_pos + seq_len]
            
            expanded_embedding = VocabParallelEmbedding(
                num_embeddings=vocab_size + num_added_tokens,
                embedding_dim=base_layer.embedding_dim,
                enable_tp=False,
                org_num_embeddings=vocab_size,
            ).to(input_ids.device)
            
            added_tokens_start_index = org_vocab_size_padded
            added_tokens_end_index = added_tokens_start_index + num_added_tokens
            expanded_embedding.weight.data[:vocab_size] = base_layer.weight.data[:vocab_size]
            expanded_embedding.weight.data[added_tokens_start_index:added_tokens_end_index] = new_embeddings[lora_id]
            result = expanded_embedding.forward(seq_input)
            
            lora_a_weights = embedding_A_buffer[lora_id]
            lora_b_weights = embedding_B_buffer[lora_id]
            scaling = batch_info.scalings[lora_id]
            
            after_a = F.embedding(seq_input, lora_a_weights.t())
            result += (after_a @ lora_b_weights.t()) * scaling
            
            expected_results.append(result)
            current_pos += seq_len
        return torch.cat(expected_results, dim=0)

    def _get_token_weight_indices_manual(self, input_ids: torch.Tensor, batch_info: LoRABatchInfo) -> torch.Tensor:
        """Manual implementation of _get_token_weight_indices for testing."""
        token_weight_indices = torch.zeros(input_ids.shape[0], dtype=torch.int32, device=input_ids.device)
        
        current_pos = 0
        for i in range(batch_info.bs):
            seg_len = batch_info.seg_lens[i]
            weight_idx = batch_info.weight_indices[i]
            token_weight_indices[current_pos:current_pos + seg_len] = weight_idx
            current_pos += seg_len
            
        return token_weight_indices

    def test_embeddings(self):
        """
        Comprehensive test for VocabParallelEmbeddingWithLoRA.
        
        Tests different configurations including:
        - Number of LoRAs (1, 2, 4)
        - Devices (CUDA, CPU)
        - Vocabulary sizes
        - Embedding dimensions
        """
        
        for device in self.devices:
            for num_loras in self.num_loras_list:
                for vocab_size in self.vocab_sizes:
                    for embed_dim in self.embed_dims:
                        with self.subTest(
                            device=device, 
                            num_loras=num_loras, 
                            vocab_size=vocab_size, 
                            embed_dim=embed_dim
                        ):
                            self._test_single_configuration(
                                device, num_loras, vocab_size, embed_dim
                            )

    def _test_single_configuration(
        self, 
        device: str, 
        num_loras: int, 
        vocab_size: int, 
        embed_dim: int
    ):
        """Test a single configuration of the LoRA embedding layer."""
        
        base_layer = self._create_base_embedding_layer(vocab_size, embed_dim, device)
        lora_layer = self._create_lora_embedding_layer(base_layer, device)
        
        new_embeddings = torch.randn(num_loras, self.num_added_tokens, embed_dim, dtype=torch.float32, device=device) * 0.1
        new_embeddings_buffer, embedding_A_buffer, embedding_B_buffer = self._create_lora_weights(
            new_embeddings, num_loras, vocab_size, embed_dim, self.num_added_tokens, device
        )
    
        lora_layer.set_lora_info(
            new_embeddings_buffer, embedding_A_buffer, embedding_B_buffer
        )
        
        for batch_size in self.batch_sizes:
            seq_lens = self.seq_lens[:batch_size]
            lora_ids: List[Optional[int]] = [i % num_loras for i in range(batch_size)]
            input_ids = self._create_test_input(
                batch_size, seq_lens, vocab_size, self.num_added_tokens, device
            )
            
            batch_info = self._create_batch_info(batch_size, seq_lens, lora_ids, device)
            lora_layer.lora_backend.set_batch_info(batch_info)
            
            with torch.no_grad():
                actual_output = lora_layer.forward(input_ids)
            
            expected_output = self._compute_expected_output(
                input_ids, base_layer, new_embeddings, embedding_A_buffer, embedding_B_buffer, batch_info, vocab_size, self.num_added_tokens
            )
            
            rtol, atol = TOLERANCES[actual_output.dtype]
            torch.testing.assert_close(actual_output,
                                   expected_output,
                                   rtol=rtol,
                                   atol=atol)

        self._test_reset_functionality(lora_layer, base_layer, vocab_size, device)

    def _test_reset_functionality(
        self, 
        lora_layer: VocabParallelEmbeddingWithLoRA, 
        base_layer: VocabParallelEmbedding, 
        vocab_size: int, 
        device: str
    ):
        """Test that resetting LoRA weights works correctly."""
        
        lora_layer.set_lora = False
        test_input = torch.randint(0, vocab_size, (10,), device=device)
        lora_layer.lora_backend.set_batch_info(None)
        
        with torch.no_grad():
            lora_output = lora_layer.forward(test_input)
            base_output = base_layer.forward(test_input)
        
        self.assertTrue(
            torch.allclose(lora_output, base_output, atol=self.tolerance),
            "LoRA layer should behave like base layer after reset"
        )

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
