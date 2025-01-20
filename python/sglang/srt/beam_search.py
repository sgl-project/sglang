from dataclasses import dataclass, field
from typing import List, Optional, Set
'''
On top of vllm beam search: 
https://github.com/vllm-project/vllm/blob/5f0ec3935a0118fee8cf2764728f765c8cc53d2a/vllm/beam_search.py'''

@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    # The tokens includes the prompt.
    last_token: int
    tokens: List[int]
    finish: bool = None
    cum_logprob: float = 0.0
    text: Optional[str] = None

    # caching
    last_req_pool_idx: int = -1 # row index
    prefix: List[int] = None
    prefix_len: int = 0 # len(req.prefix_indices)

    # dotokenizer
    vid: Optional[int] = None

    def finished(self):
        return self.finish is not None
    
    def check_prefix(self, req):
        for j, (beam_token, parent_token) in enumerate(self.tokens, req.output_ids):
            if(beam_token==parent_token and j >= len(self.prefix)):
                return False
        return True

@dataclass
class BeamSearchList:
    """The temporary status of beam search.
    """
    beam_width: int = 0
    req_pool_start_idx: int = -1 # beam_width number of sucessive indices for incompleted
    completed: List[BeamSearchSequence] = field(default_factory=list)
    incompleted: List[BeamSearchSequence] = field(default_factory=list)
    
    def empty(self):
        return len(self.completed)+len(self.incompleted) == 0


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """
    sequences: List[BeamSearchSequence]

def sort_by_beam_search_score(x: BeamSearchSequence, length_penalty: float=1.0):
    seq_len = len(x.tokens)
    if x.finished() and seq_len>1:
        seq_len -= 1
    return x.cum_logprob / (seq_len**length_penalty)
