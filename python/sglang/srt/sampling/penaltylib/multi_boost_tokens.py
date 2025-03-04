import torch
import math

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)

class BatchedMultiBoostTokensPenalizer(_BatchedPenalizer):
    """
    Multi-method boost tokens penalizer applies a probability mass boost to a specified set of tokens.
    It supports three boosting functions, selected via the boost_type parameter:

      - "linear": Linearly ramp from 0 to max_boost_fraction over ramp_tokens tokens.
      - "heaviside": No boost until ramp_tokens tokens are generated; then immediately applies max_boost_fraction.
      - "tanh": Gradually increases the boost following a tanh curve:
                effective_boost = max_boost_fraction * tanh(len_output_tokens / ramp_tokens).
                
    Each request's sampling parameters should include:
      - boosted_tokens: an iterable of token ids to boost.
      - max_boost_fraction: the maximum fraction of probability mass reserved for boosted tokens.
      - ramp_tokens: the token count threshold.
      - boost_type: the type of boosting function to use ("linear", "heaviside", or "tanh").
      
    The final probability distribution is computed as:
    
      new_prob = (1 - effective_boost) * softmax(logits) + effective_boost * uniform_boost
      
    where effective_boost is computed based on the chosen boost_type.
    """
    
    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False
    
    def _is_required(self) -> bool:
        # The penalizer is needed if any request has a non-empty boosted_tokens list and proper boost parameters.
        return any(
            bool(getattr(req.sampling_params, "boosted_tokens", []))
            and getattr(req.sampling_params, "max_boost_fraction", 0.0) > 0.0
            and getattr(req.sampling_params, "ramp_tokens", 0) > 0
            for req in self.orchestrator.reqs()
        )
    
    def _prepare(self):
        """
        Prepare a boolean mask for boosted tokens, the maximum boost fractions, and ramp thresholds.
        Also, initialize a counter to track the number of tokens generated so far.
        """
        num_reqs = len(self.orchestrator.reqs())
        vocab_size = self.orchestrator.vocab_size
        
        boost_mask = torch.zeros((num_reqs, vocab_size), dtype=torch.bool, device=self.orchestrator.device)
        max_boost_fraction = torch.zeros((num_reqs, 1), dtype=torch.float32, device=self.orchestrator.device)
        ramp_tokens = torch.zeros((num_reqs, 1), dtype=torch.float32, device=self.orchestrator.device)
        
        # Store the boost type for each request
        boost_types = []
        
        for i, req in enumerate(self.orchestrator.reqs()):
            tokens = getattr(req.sampling_params, "boosted_tokens", []) or []
            mbf = getattr(req.sampling_params, "max_boost_fraction", 0.0)
            ramp = getattr(req.sampling_params, "ramp_tokens", 0)
            boost_type = getattr(req.sampling_params, "boost_type", "linear").lower()
            
            max_boost_fraction[i, 0] = mbf
            ramp_tokens[i, 0] = float(ramp)
            boost_types.append(boost_type)
            
            for token in tokens:
                if 0 <= token < vocab_size:
                    boost_mask[i, token] = True
        
        self.boost_mask = boost_mask                    # shape: (batch_size, vocab_size)
        self.max_boost_fraction = max_boost_fraction    # shape: (batch_size, 1)
        self.ramp_tokens = ramp_tokens                  # shape: (batch_size, 1)
        self.boost_types = boost_types                  # list of boost types for each request
        # Track the number of tokens generated so far per request.
        self.len_output_tokens = torch.zeros((num_reqs, 1), dtype=torch.float32, device=self.orchestrator.device)
    
    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Increment the token counter for each new token generated.
        self.len_output_tokens += 1
    
    def _apply(self, logits: torch.Tensor):
        """
        Apply the boosting function based on the boost_type for each request:
        
          - For "linear":
              effective_boost = max_boost_fraction * clamp(len_output_tokens / ramp_tokens, max=1)
          - For "heaviside":
              effective_boost = max_boost_fraction if len_output_tokens >= ramp_tokens, else 0.
          - For "tanh":
              effective_boost = max_boost_fraction * tanh(len_output_tokens / ramp_tokens)
        
        Then, compute a new probability distribution:
        
              new_prob = (1 - effective_boost) * softmax(logits) + effective_boost * uniform_boost
        
        Finally, convert back to logits.
        """
        p = torch.softmax(logits, dim=-1)  # original probability distribution
        
        # Compute the ratio of generated tokens to ramp_tokens (avoid division by zero).
        ratio = self.len_output_tokens / torch.clamp(self.ramp_tokens, min=1e-5)
        
        # Initialize effective_boost as zeros (default)
        effective_boost = torch.zeros_like(self.max_boost_fraction)
        
        # Apply different boost functions based on each request's boost_type
        for i, boost_type in enumerate(self.boost_types):
            if boost_type == "linear":
                effective_boost[i] = self.max_boost_fraction[i] * torch.clamp(ratio[i], max=1.0)
            elif boost_type == "heaviside":
                effective_boost[i] = self.max_boost_fraction[i] * (self.len_output_tokens[i] >= self.ramp_tokens[i]).float()
            elif boost_type == "tanh":
                effective_boost[i] = self.max_boost_fraction[i] * torch.tanh(ratio[i])
            else:
                # Default to linear for unknown boost types
                effective_boost[i] = self.max_boost_fraction[i] * torch.clamp(ratio[i], max=1.0)
        
        # Build a uniform boost distribution over the boosted tokens.
        count_boost = self.boost_mask.sum(dim=1, keepdim=True).float()  # number of boosted tokens per request.
        uniform_boost = torch.zeros_like(p)
        nonzero_mask = (count_boost > 0)
        if nonzero_mask.any():
            uniform_boost[nonzero_mask] = self.boost_mask[nonzero_mask].float() / count_boost[nonzero_mask]
        
        new_p = (1 - effective_boost) * p + effective_boost * uniform_boost
        new_logits = torch.log(new_p + 1e-10)
        logits.copy_(new_logits)
    
    def _filter(self, keep_indices: torch.Tensor):
        self.boost_mask = self.boost_mask[keep_indices]
        self.max_boost_fraction = self.max_boost_fraction[keep_indices]
        self.ramp_tokens = self.ramp_tokens[keep_indices]
        self.len_output_tokens = self.len_output_tokens[keep_indices]
        self.boost_types = [self.boost_types[i] for i in keep_indices.cpu().tolist()]
    
    def _merge(self, their: "BatchedMultiBoostTokensPenalizer"):
        self.boost_mask = torch.cat([self.boost_mask, their.boost_mask], dim=0)
        self.max_boost_fraction = torch.cat([self.max_boost_fraction, their.max_boost_fraction], dim=0)
        self.ramp_tokens = torch.cat([self.ramp_tokens, their.ramp_tokens], dim=0)
        self.len_output_tokens = torch.cat([self.len_output_tokens, their.len_output_tokens], dim=0)
        self.boost_types.extend(their.boost_types) 