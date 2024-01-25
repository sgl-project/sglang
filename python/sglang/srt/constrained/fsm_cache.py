from sglang.srt.constrained.fsm import RegexFSM
from sglang.srt.constrained.tokenizer import TransformerTokenizer

_enable_memory_cache = True


class FSMCache:
    def __init__(self, tokenizer_path, tokenizer_args_dict):
        self.cache = {}
        self.outlines_tokenizer = TransformerTokenizer(
            tokenizer_path, **tokenizer_args_dict
        )

    def init_fsm(self, regex):
        if _enable_memory_cache:
            if regex not in self.cache:
                fsm = RegexFSM(regex, self.outlines_tokenizer)
                self.cache[regex] = fsm
            return self.cache[regex]

        return RegexFSM(regex, self.outlines_tokenizer)
