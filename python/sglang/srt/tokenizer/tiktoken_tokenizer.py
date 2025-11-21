import functools
import json
from typing import AbstractSet, Collection, List, Literal, Union


class TiktokenProcessor:
    def __init__(self, name: str):
        self.tokenizer = TiktokenTokenizer(name)

    def image_processor(self, image):
        return {"pixel_values": [image]}


RESERVED_TOKEN_TEXTS = [f"<|reserved_{i}|>" for i in range(3, 128)]
CONTROL_TOKEN_TEXTS = [f"<|control{i}|>" for i in range(1, 705)]


PAD = "<|pad|>"
EOS = "<|eos|>"
SEP = "<|separator|>"

DEFAULT_SPECIAL_TOKENS = [PAD, SEP, EOS]
DEFAULT_CONTROL_TOKENS = {"pad": PAD, "sep": EOS, "eos": SEP}

# default + separate each single digit
PAT_STR_B = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class TiktokenTokenizer:
    def __init__(self, tokenizer_path):
        import tiktoken
        from jinja2 import Template

        # Read the JSON
        with open(tokenizer_path, "rb") as fin:
            xtok_dict = json.load(fin)

        # Copy from train/xlm/tokenizers/tiktoken_wrapper.py::Encoding::from_xtok_dict
        mergeable_ranks = {
            bytes(item["bytes"]): item["token"] for item in xtok_dict["regular_tokens"]
        }
        special_tokens = {
            bytes(item["bytes"]).decode(): item["token"]
            for item in xtok_dict["special_tokens"]
        }
        if xtok_dict["word_split"] == "V1":
            pad_str = PAT_STR_B
        else:
            assert False, f"Unknown word_split: {xtok_dict['word_split']}"
        pad_str = xtok_dict.get("pat_str", pad_str)

        kwargs = {
            "name": tokenizer_path,
            "pat_str": pad_str,
            "mergeable_ranks": mergeable_ranks,
            "special_tokens": special_tokens,
        }
        if "default_allowed_special" in xtok_dict:
            default_allowed_special = set(
                [
                    bytes(bytes_list).decode()
                    for bytes_list in xtok_dict["default_allowed_special"]
                ]
            )
        if "vocab_size" in xtok_dict:
            kwargs["explicit_n_vocab"] = xtok_dict["vocab_size"]

        # Copy from train/xlm/tokenizers/tiktoken_wrapper.py::Encoding::__init__
        default_allowed_special = None
        control_tokens = DEFAULT_CONTROL_TOKENS
        tokenizer = tiktoken.Encoding(**kwargs)
        tokenizer._default_allowed_special = default_allowed_special or set()
        tokenizer._control_tokens = control_tokens

        def encode_patched(
            self,
            text: str,
            *,
            allowed_special: Union[
                Literal["all"], AbstractSet[str]
            ] = set(),  # noqa: B006
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        ) -> List[int]:
            if isinstance(allowed_special, set):
                allowed_special |= self._default_allowed_special
            return tiktoken.Encoding.encode(
                self,
                text,
                allowed_special=allowed_special,
                disallowed_special=(),
            )

        tokenizer.encode = functools.partial(encode_patched, tokenizer)

        # Allow more tokens to prevent crash
        tokenizer._default_allowed_special |= set(DEFAULT_CONTROL_TOKENS.values())
        tokenizer._default_allowed_special |= set(
            CONTROL_TOKEN_TEXTS + RESERVED_TOKEN_TEXTS
        )

        # Convert to HF interface
        self.tokenizer = tokenizer
        self.bos_token_id = None
        self.eos_token_id = tokenizer._special_tokens[EOS]
        self.vocab_size = tokenizer.n_vocab
        self.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'system' %}{{ 'System: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: '  + message['content'] + '<|separator|>\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        self.chat_template_jinja = Template(self.chat_template)
        self.additional_stop_token_ids = None

    def encode(self, x, add_special_tokens=False):
        return self.tokenizer.encode(x)

    def decode(self, x, *args, **kwargs):
        return self.tokenizer.decode(x)

    def batch_decode(
        self, batch, skip_special_tokens=True, spaces_between_special_tokens=False
    ):
        if len(batch) > 0 and isinstance(batch[0], int):
            batch = [[x] for x in batch]
        return self.tokenizer.decode_batch(batch)

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        tools=None,
        reasoning_effort=None,
        **kwargs,  # Accept additional parameters (e.g., return_dict) for compatibility
    ):
        ret = self.chat_template_jinja.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
        return self.encode(ret) if tokenize else ret

    def __call__(self, text: List[str], **kwargs):
        return {
            "input_ids": [self.encode(x) for x in text],
        }

    def init_xgrammar(self):
        from xgrammar import TokenizerInfo

        XGRAMMAR_SPECIAL_TOKEN_TEMPLATE = "<|xg_special_token_{}|>"

        enc = self.tokenizer
        encoded_vocab = {**enc._mergeable_ranks, **enc._special_tokens}
        encoded_vocab = [
            token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])
        ]
        override_stop_tokens = [2]  # eos
        # These are treated as special tokens in xgrammar; we want to avoid them
        # For now, xgrammar treats anything starting with b'\x00' as a special token
        xgrammar_special_token_ids = []
        for i, token in enumerate(encoded_vocab):
            if isinstance(token, bytes) and token.startswith(b"\x00"):
                xgrammar_special_token_ids.append(i)

        for i, id in enumerate(xgrammar_special_token_ids):
            encoded_vocab[id] = XGRAMMAR_SPECIAL_TOKEN_TEMPLATE.format(i)
        tokenizer_info = TokenizerInfo(
            encoded_vocab, stop_token_ids=override_stop_tokens
        )
        assert len(tokenizer_info.special_token_ids) == 0

        return tokenizer_info, override_stop_tokens
