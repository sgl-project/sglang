"""Cache for the compressed finite state machine."""

from sglang.srt.constrained import RegexGuide, TransformerTokenizer
from sglang.srt.constrained.base_cache import BaseCache


class FSMCache(BaseCache):
    def __init__(self, tokenizer_path, tokenizer_args_dict, enable=True):
        super().__init__(enable=enable)

        if tokenizer_path.endswith(".json") or tokenizer_path.endswith(".model"):
            # Do not support TiktokenTokenizer or SentencePieceTokenizer
            return

        from importlib.metadata import version

        if version("outlines") >= "0.0.35":
            from transformers import AutoTokenizer

            tokenizer_args_dict.setdefault("padding_side", "left")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, **tokenizer_args_dict
            )
            try:
                self.outlines_tokenizer = TransformerTokenizer(tokenizer)
            except AttributeError:
                # FIXME: tmp fix for chatglm2 & chatglm3 (pad_token_id=0)
                origin_pad_token_id = tokenizer.pad_token_id

                def fset(self, value):
                    self._value = value

                type(tokenizer).pad_token_id = property(
                    fget=type(tokenizer).pad_token_id.fget, fset=fset
                )
                self.outlines_tokenizer = TransformerTokenizer(tokenizer)
                self.outlines_tokenizer.tokenizer.pad_token_id = origin_pad_token_id
                self.outlines_tokenizer.pad_token_id = origin_pad_token_id
                self.outlines_tokenizer.pad_token = (
                    self.outlines_tokenizer.tokenizer.pad_token
                )
                self.outlines_tokenizer.vocabulary = (
                    self.outlines_tokenizer.tokenizer.get_vocab()
                )
        else:
            self.outlines_tokenizer = TransformerTokenizer(
                tokenizer_path, **tokenizer_args_dict
            )

    def init_value(self, regex):
        return RegexGuide(regex, self.outlines_tokenizer)
