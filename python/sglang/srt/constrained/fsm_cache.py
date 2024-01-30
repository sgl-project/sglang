import time

from sglang.srt.constrained.fsm import RegexFSM
from sglang.srt.constrained.tokenizer import TransformerTokenizer

_enable_memory_cache = True


class FSMCache:
    def __init__(self, tokenizer_path, tokenizer_args_dict):
        self.reset()
        self.outlines_tokenizer = TransformerTokenizer(
            tokenizer_path, **tokenizer_args_dict
        )

    def reset(self):
        self.cache = {}
        self.metrics = {"total": 0, "hit": 0, "avg_init_time": 0}

    def init_fsm(self, regex):
        def _init_fsm_helper(regex):
            start = time.monotonic()
            fsm = RegexFSM(regex, self.outlines_tokenizer)
            init_time = time.monotonic() - start
            curr_total = self.metrics["total"]
            new_total = curr_total + 1

            # Update average init time without old_avg * old_total to avoid overflow.
            self.metrics["avg_init_time"] = (init_time / new_total) + (
                curr_total / new_total
            ) * self.metrics["avg_init_time"]
            self.metrics["total"] += 1
            return fsm

        if _enable_memory_cache:
            if regex not in self.cache:
                self.cache[regex] = _init_fsm_helper(regex)
            else:
                self.metrics["hit"] += 1
            fsm = self.cache[regex]
        else:
            fsm = _init_fsm_helper(regex)
        return fsm

    def get_cache_hit_rate(self):
        if self.metrics["total"] == 0:
            return 0
        return self.metrics["hit"] / self.metrics["total"]

    def get_avg_init_time(self):
        return self.metrics["avg_init_time"]
