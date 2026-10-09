# Adapted from https://huggingface.co/stepfun-ai/GOT-OCR2_0

from transformers import Qwen2Config

# `<|im_end|>`, which closes every turn of GOT's conversation template. Distinct
# from the config's `im_end_token` (151858 `</img>`), which closes an image span.
CHAT_EOS_TOKEN_ID = 151645


class GOTConfig(Qwen2Config):
    """Config for `stepfun-ai/GOT-OCR2_0` (`model_type: GOT`).

    The checkpoint's `auto_map` sends `AutoConfig` into `modeling_GOT.py`, which
    transformers refuses to import: its static scan sees `import verovio`, a
    sheet-music renderer that file imports inside one branch inference never
    reaches. Declaring the fields here builds the config without executing that
    file, which keeps `--trust-remote-code` free for the tiktoken tokenizer.
    """

    model_type = "GOT"

    def __init__(
        self,
        image_token_len: int = 256,
        im_start_token: int = 151857,
        im_end_token: int = 151858,
        im_patch_token: int = 151859,
        use_im_start_end: bool = True,
        **kwargs,
    ) -> None:
        self.image_token_len = image_token_len
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.im_patch_token = im_patch_token
        self.use_im_start_end = use_im_start_end
        super().__init__(**kwargs)

        # The checkpoint kept its base model's eos (`<|endoftext|>`) and never
        # listed the chat terminator, so decoding would run to max_new_tokens
        # emitting `<|im_end|>`. Qwen2 chat configs list both.
        eos_ids = self.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        if CHAT_EOS_TOKEN_ID not in eos_ids:
            eos_ids.append(CHAT_EOS_TOKEN_ID)
        self.eos_token_id = eos_ids
