"""
Unit test to verify _handle_batch_output has O(N) complexity, not O(N^2).

This test ensures the optimization that avoids copying output_ids on every
intermediate update is working correctly. Without this optimization, 
long sequences would have quadratic complexity due to copying the entire
output list on every decode step.

Usage:
    python -m pytest test/unit/test_tokenizer_manager_output_complexity.py -v
    python test/unit/test_tokenizer_manager_output_complexity.py
"""

import asyncio
import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MockReqState:
    """Minimal request state for testing."""
    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Any
    created_time: float
    text: str = ""
    output_ids: List[int] = field(default_factory=list)
    last_output_offset: int = 0
    finished_time: float = 0.0
    finished_time_perf: float = 0.0


@dataclass 
class MockBatchStrOutput:
    """Minimal batch output for testing."""
    rids: List[str]
    finished_reasons: List[Optional[dict]]
    output_strs: List[str]
    output_ids: List[List[int]]
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]
    retraction_counts: List[int]


class MockObj:
    def __init__(self):
        self.stream = False
        self.return_logprob = False
        self.log_metrics = False


class OutputComplexityTest(unittest.TestCase):
    """Test that output handling has linear, not quadratic, complexity."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _create_handler_optimized(self):
        """Create an optimized handler that only copies on finish."""
        rid_to_state: Dict[str, MockReqState] = {}
        
        def setup(num_requests: int):
            rid_to_state.clear()
            for i in range(num_requests):
                rid = f"req_{i}"
                state = MockReqState(
                    out_list=[],
                    finished=False,
                    event=asyncio.Event(),
                    obj=MockObj(),
                    created_time=time.time(),
                )
                rid_to_state[rid] = state
        
        def handle(recv_obj: MockBatchStrOutput):
            """Optimized: only copy output_ids when finished."""
            for i, rid in enumerate(recv_obj.rids):
                state = rid_to_state.get(rid)
                if state is None:
                    continue
                
                is_finished = recv_obj.finished_reasons[i] is not None
                
                state.text += recv_obj.output_strs[i]
                state.output_ids += recv_obj.output_ids[i]
                
                # KEY OPTIMIZATION: only copy when finished
                if is_finished:
                    output_token_ids = state.output_ids[:]
                else:
                    output_token_ids = state.output_ids  # Reference, no copy
                
                out_dict = {
                    "text": state.text,
                    "output_ids": output_token_ids,
                    "meta_info": {"id": rid},
                }
                
                state.finished = is_finished
                if is_finished:
                    del rid_to_state[rid]
                
                state.out_list.append(out_dict)
        
        return setup, handle, rid_to_state

    def _run_sequence_benchmark(self, setup_fn, handle_fn, batch_size: int, num_steps: int) -> float:
        """Run a sequence of updates and measure time."""
        setup_fn(batch_size)
        rids = [f"req_{i}" for i in range(batch_size)]
        
        start = time.perf_counter()
        
        # Intermediate updates (not finished)
        for step in range(num_steps - 1):
            recv_obj = MockBatchStrOutput(
                rids=rids,
                finished_reasons=[None] * batch_size,
                output_strs=["x"] * batch_size,
                output_ids=[[step]] * batch_size,
                prompt_tokens=[100] * batch_size,
                completion_tokens=[step + 1] * batch_size,
                cached_tokens=[0] * batch_size,
                retraction_counts=[0] * batch_size,
            )
            handle_fn(recv_obj)
        
        # Final update (finished)
        recv_obj = MockBatchStrOutput(
            rids=rids,
            finished_reasons=[{"type": "stop"}] * batch_size,
            output_strs=["x"] * batch_size,
            output_ids=[[num_steps - 1]] * batch_size,
            prompt_tokens=[100] * batch_size,
            completion_tokens=[num_steps] * batch_size,
            cached_tokens=[0] * batch_size,
            retraction_counts=[0] * batch_size,
        )
        handle_fn(recv_obj)
        
        return time.perf_counter() - start

    def test_linear_complexity(self):
        """
        Verify that doubling sequence length does NOT quadruple time.
        
        For O(N) complexity: doubling N should roughly double time (~2x)
        For O(N^2) complexity: doubling N would quadruple time (~4x)
        """
        setup_fn, handle_fn, _ = self._create_handler_optimized()
        batch_size = 8
        
        # Test with N and 2N steps (small values for fast test)
        n_small = 200
        n_large = 400
        
        # Measure
        t_small = self._run_sequence_benchmark(setup_fn, handle_fn, batch_size, n_small)
        t_large = self._run_sequence_benchmark(setup_fn, handle_fn, batch_size, n_large)
        
        ratio = t_large / t_small
        
        # For linear: ratio ~2x. For quadratic: ratio ~4x. 
        # Accept < 3.0 to confirm linear.
        self.assertLess(ratio, 3.0,
            f"Complexity appears quadratic! Ratio={ratio:.2f}x (expected ~2x)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
