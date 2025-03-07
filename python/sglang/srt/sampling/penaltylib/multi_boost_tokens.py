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
        print('is_required: ', any(
            bool(getattr(req.sampling_params, "boosted_tokens", []))
            and getattr(req.sampling_params, "max_boost_fraction", 0.0) > 0.0
            and getattr(req.sampling_params, "ramp_tokens", 0) > 0
            for req in self.orchestrator.reqs()
        ))  
        # print the sampling_params for each request
        for req in self.orchestrator.reqs():
            print('sampling_params: ', f"boosted_tokens: {req.sampling_params.boosted_tokens}, max_boost_fraction: {req.sampling_params.max_boost_fraction}, ramp_tokens: {req.sampling_params.ramp_tokens}")
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
        self.boost_mask = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )
        for i,req in enumerate(self.orchestrator.reqs()):
            for token in req.sampling_params.boosted_tokens:
                self.boost_mask[i, token] = 1
        
        self.ramp_scale = torch.tensor([req.sampling_params.ramp_tokens for req in self.orchestrator.reqs()], device=self.orchestrator.device)
        self.boost_factor = torch.tensor([req.sampling_params.max_boost_fraction for req in self.orchestrator.reqs()], device=self.orchestrator.device)
        self.len_output_tokens = torch.zeros(
            size=(len(self.orchestrator.reqs()),),
            dtype=torch.int32,
            device=self.orchestrator.device,
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Increment the token counter for each new token generated.
        self.len_output_tokens += 1
    
    def _apply(self, logits: torch.Tensor):
        """
        Apply the boosting function based on the boost_type for each request.
        """
        print('logits shape: ', logits.shape)
        print('max logit: ', logits[0].max())
        print('min logit: ', logits[0].min())
        print('logit before: ', logits[0,:20])

        
        
        # Compute the ratio of generated tokens to ramp_tokens (avoid division by zero).
        ratio = self.len_output_tokens / self.ramp_scale # element-wise division, shape: (B)

        print('ratio shape: ', ratio.shape)
        
        # Initialize effective_boost as zeros
        effective_boost = torch.zeros_like(logits) # shape: (B, V)
        
        # Compute all boost types in parallel
        for i,req in enumerate(self.orchestrator.reqs()):
            for token in req.sampling_params.boosted_tokens:
                if req.sampling_params.boost_type == 'linear':
                    effective_boost[i, token] = self.boost_factor[i] * torch.clamp(ratio[i], max=1.0)
                elif req.sampling_params.boost_type == 'heaviside':
                    effective_boost[i, token] = self.boost_factor[i] * (self.len_output_tokens[i] >= self.ramp_scale[i]).float()
                elif req.sampling_params.boost_type == 'tanh':
                    effective_boost[i, token] = self.boost_factor[i] * torch.tanh(ratio[i])
                else:
                    effective_boost[i, token] = torch.zeros_like(ratio[i])
        
        

        # Combine results using masks
        print('effective_boost shape: ', effective_boost.shape)
        print('effective_boost: ', effective_boost)
        # use formula Adjusted logits = logits + effective_boost 
        # where effective_boost is the boost for each token
        # namely boost is 0 if the token is not boosted
        print('boost_mask shape: ', self.boost_mask.shape)
        print('boost_mask: ', self.boost_mask)
        logits.add_(effective_boost)
        print('max logit: ', logits[0].max())
        print('min logit: ', logits[0].min())
        print('logit after: ', logits[0,:20])

    
    def _filter(self, keep_indices: torch.Tensor):
        self.boost_mask = self.boost_mask[keep_indices]
        self.ramp_scale = self.ramp_scale[keep_indices]
        self.boost_factor = self.boost_factor[keep_indices]
        self.len_output_tokens = self.len_output_tokens[keep_indices]

    def _merge(self, their: "BatchedMultiBoostTokensPenalizer"):
        print(f"{self.boost_mask.shape=}, {their.boost_mask.shape=}")
        self.boost_mask = torch.cat([self.boost_mask, their.boost_mask], dim=0)
        self.ramp_scale = torch.cat([self.ramp_scale, their.ramp_scale], dim=0)
        self.boost_factor = torch.cat([self.boost_factor, their.boost_factor], dim=0)
        self.len_output_tokens = torch.cat([self.len_output_tokens, their.len_output_tokens], dim=0)
        