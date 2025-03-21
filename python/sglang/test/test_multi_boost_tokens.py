import unittest
import torch
import numpy as np

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.python.sglang.srt.sampling.penaltylib.reasoning_penalty import BatchedMultiBoostTokensPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator


class MockRequest:
    """Mock request class for testing"""
    def __init__(self, sampling_params):
        self.sampling_params = sampling_params


class MockScheduleBatch:
    """Mock schedule batch class for testing"""
    def __init__(self, reqs):
        self.reqs = reqs
        self.device = "cpu"


class TestBatchedMultiBoostTokensPenalizer(unittest.TestCase):
    def test_boosting_linear(self):
        # Create test sampling parameters with boosted tokens
        vocab_size = 10
        tokens_to_boost = [1, 3, 5]  # We'll boost tokens 1, 3, and 5
        max_boost = 0.5
        ramp_tokens = 2
        
        params = SamplingParams(
            boosted_tokens=tokens_to_boost,
            max_boost_fraction=max_boost,
            ramp_tokens=ramp_tokens,
            boost_type="linear"
        )
        
        # Create a mock request with these parameters
        req = MockRequest(params)
        
        # Create a mock batch with the request
        batch = MockScheduleBatch([req])
        
        # Create the orchestrator
        orchestrator = BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={BatchedMultiBoostTokensPenalizer}
        )
        
        # Create logits
        logits = torch.tensor([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Equal logits for all tokens
        ], dtype=torch.float32)
        
        probs_before = torch.softmax(logits.clone(), dim=-1)
        
        # Apply the penalizer once (should have 1/2 effect due to ramp=2)
        orchestrator.cumulate_output_tokens(torch.tensor([[0]]))
        orchestrator.apply(logits)
        
        probs_1token = torch.softmax(logits.clone(), dim=-1)
        
        # Check that the probability of boosted tokens increased
        boosted_sum_1token = probs_1token[0, tokens_to_boost].sum().item()
        expected_ratio_1token = (1 - 0.5 * 0.5) * 0.3 + 0.5 * 0.5  # (1-effective_boost)*original + effective_boost*uniform
        self.assertAlmostEqual(boosted_sum_1token, expected_ratio_1token, places=5)
        
        # Apply the penalizer again (should have full effect now)
        orchestrator.cumulate_output_tokens(torch.tensor([[0]]))
        
        # Reset logits
        logits = torch.tensor([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Equal logits for all tokens
        ], dtype=torch.float32)
        
        orchestrator.apply(logits)
        
        probs_2tokens = torch.softmax(logits.clone(), dim=-1)
        
        # Check that the probability of boosted tokens increased to the maximum
        boosted_sum_2tokens = probs_2tokens[0, tokens_to_boost].sum().item()
        expected_ratio_2tokens = (1 - 0.5) * 0.3 + 0.5  # (1-max_boost)*original + max_boost
        self.assertAlmostEqual(boosted_sum_2tokens, expected_ratio_2tokens, places=5)

    def test_different_boost_types(self):
        # Test heaviside and tanh boost types
        vocab_size = 10
        tokens_to_boost = [1, 3, 5]
        max_boost = 0.5
        ramp_tokens = 2
        
        for boost_type in ["heaviside", "tanh"]:
            params = SamplingParams(
                boosted_tokens=tokens_to_boost,
                max_boost_fraction=max_boost,
                ramp_tokens=ramp_tokens,
                boost_type=boost_type
            )
            
            req = MockRequest(params)
            batch = MockScheduleBatch([req])
            
            orchestrator = BatchedPenalizerOrchestrator(
                vocab_size=vocab_size,
                batch=batch,
                penalizers={BatchedMultiBoostTokensPenalizer}
            )
            
            # Create logits
            logits = torch.tensor([
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Equal logits
            ], dtype=torch.float32)
            
            # Apply the penalizer once
            orchestrator.cumulate_output_tokens(torch.tensor([[0]]))
            orchestrator.apply(logits.clone())
            
            # Apply the penalizer a second time
            orchestrator.cumulate_output_tokens(torch.tensor([[0]]))
            
            # Reset logits
            logits = torch.tensor([
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Equal logits
            ], dtype=torch.float32)
            
            orchestrator.apply(logits)
            
            # Verify that the probabilities are modified (we don't test exact values here,
            # just that they're different from uniform)
            probs = torch.softmax(logits, dim=-1)
            self.assertTrue(torch.any(probs != 0.1).item())


if __name__ == "__main__":
    unittest.main() 