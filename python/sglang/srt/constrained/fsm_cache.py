import threading

from sglang.srt.constrained.fsm import RegexFSM
from sglang.srt.constrained.tokenizer import TransformerTokenizer


def get_fsm(regex, tokenizer, fsm_cache_entry):
    outlines_tokenizer = TransformerTokenizer(tokenizer)
    fsm = RegexFSM(regex, outlines_tokenizer)
    fsm_cache_entry.fsm = fsm
    fsm_cache_entry.event.set()


class FSMCacheEntry:
    def __init__(self):
        self.fsm = None
        self.event = threading.Event()


class FSMCache:
    def __init__(self, tokenizer):
        self.cache = {}
        self.tokenizer = tokenizer

    def init_fsm_in_background(self, regex):
        if regex not in self.cache:
            self.cache[regex] = FSMCacheEntry()
            threading.Thread(
                target=get_fsm,
                args=(
                    regex,
                    self.tokenizer,
                    self.cache[regex],
                ),
            ).start()

    def get_fsm(self, regex):
        self.init_fsm_in_background(regex)
        entry = self.cache[regex]
        entry.event.wait()
        return entry.fsm
