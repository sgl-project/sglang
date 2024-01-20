import os

from diskcache import Cache

from sglang.srt.constrained.fsm import RegexFSM
from sglang.srt.constrained.tokenizer import TransformerTokenizer

home_dir = os.path.expanduser("~")
cache_dir = os.environ.get("SGLANG_CACHE_DIR", f"{home_dir}/.cache/sglang")
disk_fsm_cache = Cache(cache_dir, eviction_policy="none", cull_limit=0)
_enable_disk_cache = True


class FSMCache:
    def __init__(self, tokenizer_path, tokenizer_args_dict):
        self.cache = {}
        self.outlines_tokenizer = TransformerTokenizer(
            tokenizer_path, **tokenizer_args_dict
        )

    def init_fsm(self, regex):
        if regex not in self.cache:
            fsm = RegexFSM(regex, self.outlines_tokenizer)
            self.cache[regex] = fsm

        return self.cache[regex]
