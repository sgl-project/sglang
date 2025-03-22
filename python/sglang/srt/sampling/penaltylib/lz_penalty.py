import torch
import math

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)

def compute_row_encoding_cost(x: torch.Tensor) -> torch.Tensor:
    """
    Computes, for each row, the minimum over columns of:
      (log(n - i + 1) + log(x_i)) / x_i
    for valid x_i > 0.
    This is a variation of the LZSS / LZ77 sliding window matching algorithm.
    Works for both 2D and higher dimensional tensors, operating on the last dimension.
    """
    n = x.size(-1)
    indices = torch.arange(n, dtype=torch.float, device=x.device)
    # Add singleton dimensions to broadcast correctly for both 2D and higher dims
    log_term = torch.log2(n - indices + 1)
    log_term = log_term.view(*([1] * (x.dim() - 1)), -1)

    x = x.float()
    
    # Create a mask for valid x_i > 0
    mask = x > 0
    # Compute log(x) only where valid, set invalid ones to inf (since we take min)
    log_x = torch.where(mask, torch.log2(x), torch.full_like(x, float('inf')))
    
    # Elementwise compute (log_term + log_x) / x.
    # For invalid entries, set to inf.
    value = torch.where(mask, (log_term + log_x) / x, torch.full_like(x, float('inf')))
    
    # Return the minimum value along last dimension
    return torch.min(value, dim=-1).values


def compute_reversed_consecutive_counts(E: torch.BoolTensor) -> torch.Tensor:
    '''
    Compute the number of consecutive ones from the right.
    '''
    E_rev = E.flip(dims=[1]).float()
    cp = torch.cumprod(E_rev, dim=1)
    counts = cp.sum(dim=1)
    return counts.int()  # shape: (num_windows,)


def compute_ngram_penalty_stateless(W_context: torch.Tensor, 
                                    G_vocab: torch.Tensor, 
                                    n: int) -> torch.Tensor:
    '''
    Compute the n-gram penalty for a single request.
    '''
    w = W_context.size(0)  # context length
    if w <= n:
        # If context is too short, return zero penalty.
        return torch.zeros(G_vocab, device=W_context.device)

    # Extract sliding windows: there are (w - n + 1) windows; drop the last one.
    windows = W_context[:-1].unfold(0, n, 1)    # shape: (w - n  - 1 , n)

    # Define 
    L_lookback = W_context[-n:] # shape: (n,)
    
    # Compare each window with the lookback.
    E = windows == L_lookback.unsqueeze(0) # shape: (w - n - 1, n)
    
    # Compute potential U for each window.
    U = compute_reversed_consecutive_counts(E)
    
    # Compute match matrix using tokens from indices n to w-1 (i.e. excluding the last token).
    match_context = W_context[n:]
    M = (G_vocab.unsqueeze(1) == match_context.unsqueeze(0)).int()
    RP = (M * U) + M
    log_vocab = math.log2(len(G_vocab))
    RP = torch.clamp(compute_row_encoding_cost(RP), min=0.0, max=log_vocab)
    return log_vocab - RP


def compute_ngram_penalty_naive(W_context: torch.Tensor, 
                                    G_vocab: torch.Tensor, 
                                    Ns: int
                                    ) -> torch.Tensor:
    '''
    Compute the n-gram penalty for each request in the batch.
    '''
    batch_size = W_context.size(0)
    penalties = torch.zeros(batch_size, device=W_context.device)
    for i in range(batch_size):
        n = Ns[i]
        penalties[i] = compute_ngram_penalty_stateless(W_context[i], G_vocab, n)
    return penalties

def compute_ngram_penalty_batched(W_context: torch.Tensor, 
                                    G_vocab: torch.Tensor, 
                                    n: torch.Tensor
                                    ) -> torch.Tensor:
    '''
    Compute the n-gram penalty for each request in the batch.
    Implemented using vectorized operations.
    '''
    w = W_context.size(1)
    if w <= n:
        # If context is too short, return zero penalty.
        return torch.zeros_like(G_vocab, device=W_context.device)

    windows = W_context[:,:-1].unfold(1, n, 1)
    lookback = W_context[:,-n:]
    E = windows.transpose(1,2) == lookback.unsqueeze(-1)
    U = compute_reversed_consecutive_counts(E)
    match_context = W_context[:,n:]
    M = (G_vocab.view(1,1,-1) == match_context.unsqueeze(-1)).int()
    M = M.permute(2,0,1)
    K = (M * U) + M
    log_vocab = math.log2(len(G_vocab))
    RP = torch.clamp(compute_row_encoding_cost(K), min=0.0, max=log_vocab).t()
    return log_vocab - RP
    

class BatchedLZPenalizer(_BatchedPenalizer):
    """
    LZ penalizer penalizes tokens based on their presence in n-gram windows.
    The penalty is inspired by the LZSS / LZ77 sliding window matching algorithm,
    and is infact proportional to the change in codelength under universal comperssion
    associated with the token.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.ngram_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        # Store ngram parameters for each request
        self.ngram_penalties = torch.tensor(
            [req.sampling_params.ngram_penalty for req in self.orchestrator.reqs()],
            dtype=torch.float32,
            device=self.orchestrator.device
        )
        self.ngram_ns = torch.tensor(
            [req.sampling_params.ngram_n for req in self.orchestrator.reqs()],
            dtype=torch.long,
            device=self.orchestrator.device
        )

        self.ngram_windows = torch.tensor(
            [req.sampling_params.ngram_lookback_window for req in self.orchestrator.reqs()],
            dtype=torch.long,
            device=self.orchestrator.device
        )

        # check that ngram_windows and ngrams are the same for each request
        # if not, we can support this but further optimization is needed
        # to pad the context and lookbacks during the penalty computation
        self._batched = torch.all(
            self.ngram_windows == self.ngram_windows[0]) and torch.all(
                self.ngram_ns == self.ngram_ns[0])


        self.Ws = []
        for req in self.orchestrator.reqs():
            # Initialize empty state for each request
            self.Ws.append(
                torch.empty(0, dtype=torch.long, device=self.orchestrator.device))
        if self._batched:
            self.Ws = torch.cat(self.Ws, dim=0).int()

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        if self._batched:
            w = self.Ws[0].size(0)
            winlen = self.ngram_windows[0]
            if w >= winlen:
                self.Ws = torch.cat([self.Ws[:, 1:], output_ids.unsqueeze(1)], dim=1)
            else:
                self.Ws = torch.cat([self.Ws, output_ids.unsqueeze(1)], dim=1)
        else:
            for i, _W in enumerate(self.Ws):
                winlen = self.ngram_windows[i]
                w = _W.size(0)
                if w >= winlen:
                    self.Ws[i] = torch.cat([_W[1:], output_ids[i].unsqueeze(0)], dim=0)
                else:
                    self.Ws[i] = torch.cat([_W, output_ids[i].unsqueeze(0)], dim=0)
        

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # Apply penalties for each request
        G_vocab = torch.arange(self.orchestrator.vocab_size, dtype=torch.long, device=self.orchestrator.device)
        
        # compute the penalty for each request
        if self._batched:
            penalties = compute_ngram_penalty_batched(self.Ws, G_vocab, self.ngram_ns[0])
        else:
            penalties = compute_ngram_penalty_naive(self.Ws, G_vocab, self.ngram_ns)
        
        # scale the penalties by the user-specified penalty weights
        scaled_penalties = penalties * self.ngram_penalties.unsqueeze(1)
        
        # apply the penalty to the logits
        logits.sub_(scaled_penalties)

        return logits

    def _filter(self, keep_indices: torch.Tensor):
        # Update Ws tensor by selecting rows based on keep_indices
        self.Ws = self.Ws[keep_indices]
        # Update tensors of parameters
        self.ngram_penalties = self.ngram_penalties[keep_indices]
        self.ngram_ns = self.ngram_ns[keep_indices]
        self.ngram_windows = self.ngram_windows[keep_indices]

    def _merge(self, their: "BatchedLZPenalizer"):
        # Merge parameter tensors
        self.ngram_penalties = torch.cat([self.ngram_penalties, their.ngram_penalties], dim=0)
        self.ngram_ns = torch.cat([self.ngram_ns, their.ngram_ns], dim=0)
        self.ngram_windows = torch.cat([self.ngram_windows, their.ngram_windows], dim=0)
        # need to check the batched flag is the same for both
        self._batched = torch.all(
            self.ngram_windows == self.ngram_windows[0]) and torch.all(
                self.ngram_ns == self.ngram_ns[0])

        if self._batched:
            self.Ws = torch.cat([self.Ws, their.Ws], dim=0)
        else:
            Ws = []
            for i, _W in enumerate(self.Ws):
                Ws.append(_W)
            for i, _W in enumerate(their.Ws):
                Ws.append(_W)
            self.Ws = Ws



if __name__ == "__main__":
    # Debug tests to compare batch and stateless implementation
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test case 1: Single sequence
    print("\n=== Test Case 1: Single Sequence ===")
    # Create a simple context with some repetition
    context = torch.tensor([1, 2, 3, 4, 2, 3, 4, 5], dtype=torch.long, device=device)
    vocab = torch.arange(10, dtype=torch.long, device=device)
    n = 3
    
    # Compute stateless penalty
    stateless_result = compute_ngram_penalty_stateless(context, vocab, n)
    print(f"Stateless result: {stateless_result}")
    
    # Setup for batched computation
    batched_context = context.unsqueeze(0)  # Add batch dimension
    batched_n = torch.tensor([n], dtype=torch.long, device=device)
    
    # Compute batched penalty
    batched_result = compute_ngram_penalty_batched(batched_context, vocab, batched_n)
    print(f"Batched result: {batched_result.squeeze()}")
    
    # Check if results match
    match = torch.allclose(stateless_result, batched_result.squeeze(), rtol=1e-5)
    print(f"Results match: {match}")
    
    # Test case 2: Multiple sequences in batch
    print("\n=== Test Case 2: Multiple Sequences ===")
    # Create a batch of contexts
    batch_size = 3
    contexts = [
        torch.tensor([1, 2, 3, 4, 1, 2, 3, 5], dtype=torch.long, device=device),
        torch.tensor([5, 6, 7, 8, 5, 6, 7, 9], dtype=torch.long, device=device),
        torch.tensor([9, 8, 7, 6, 9, 8, 7, 5], dtype=torch.long, device=device)
    ]
    
    # Pad to same length if needed (not needed in this example since all are same length)
    batched_contexts = torch.stack(contexts)
    
    # Different n-gram sizes for each sequence
    ns = torch.tensor([2, 3, 4], dtype=torch.long, device=device)
    
    # Compute batched penalties
    batched_results = compute_ngram_penalty_batched(batched_contexts, vocab, ns)
    print(f"Batched results shape: {batched_results.shape}")
    
    # Compare with individual stateless calculations
    for i in range(batch_size):
        stateless_result = compute_ngram_penalty_stateless(contexts[i], vocab, ns[i].item())
        print(f"Sequence {i+1}:")
        print(f"  Stateless: {stateless_result}")
        print(f"  Batched: {batched_results[i]}")
        match = torch.allclose(stateless_result, batched_results[i], rtol=1e-5)
        print(f"  Match: {match}")
    
    # Test case 3: Edge cases
    print("\n=== Test Case 3: Edge Cases ===")
    
    # Short context (smaller than n)
    short_context = torch.tensor([1, 2], dtype=torch.long, device=device)
    n_large = 3
    
    stateless_result = compute_ngram_penalty_stateless(short_context, vocab, n_large)
    print(f"Short context (stateless): {stateless_result}")
    
    # Empty context
    empty_context = torch.tensor([], dtype=torch.long, device=device)
    try:
        stateless_result = compute_ngram_penalty_stateless(empty_context, vocab, 1)
        print(f"Empty context (stateless): {stateless_result}")
    except Exception as e:
        print(f"Empty context error (expected): {e}")
    
    # Test case 4: Compare with naive batch implementation
    print("\n=== Test Case 4: Compare with Naive Batch Implementation ===")
    contexts = torch.stack([
        torch.tensor([1, 2, 3, 4, 5, 1, 2, 3], dtype=torch.long, device=device),
        torch.tensor([5, 6, 7, 8, 9, 5, 6, 7], dtype=torch.long, device=device)
    ])
    n_fixed = 2
    
    # Use naive batch implementation (loop-based)
    naive_results = compute_ngram_penalty_naive(contexts, vocab, n_fixed)
    print(f"Naive batched results: {naive_results}")
    
    # Use vectorized batch implementation
    batched_n = torch.tensor([n_fixed, n_fixed], dtype=torch.long, device=device)
    vectorized_results = compute_ngram_penalty_batched(contexts, vocab, batched_n)
    print(f"Vectorized batched results: {vectorized_results}")
    
    # Check if results match
    match = torch.allclose(naive_results, vectorized_results, rtol=1e-5)
    print(f"Implementations match: {match}")
    
    print("\nAll tests completed.")