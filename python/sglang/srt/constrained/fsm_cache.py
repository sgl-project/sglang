from sglang.srt.constrained.base_cache import BaseCache
from sglang.srt.constrained.fsm import RegexFSM
from sglang.srt.constrained.tokenizer import TransformerTokenizer


class FSMCache(BaseCache):
    def __init__(self, tokenizer_path, tokenizer_args_dict, enable=True):
        super().__init__(enable=enable)
        self.outlines_tokenizer = TransformerTokenizer(
            tokenizer_path, **tokenizer_args_dict
        )

    def init_value(self, regex):
        return RegexFSM(regex, self.outlines_tokenizer)
