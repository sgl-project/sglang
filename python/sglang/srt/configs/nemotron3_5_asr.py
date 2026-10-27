from transformers import Nemotron3_5AsrConfig as HFNemotron3_5AsrConfig


class Nemotron3_5AsrConfig(HFNemotron3_5AsrConfig):
    """Nemotron 3.5 ASR config with the text-model fields the engine reads.

    The RNN-T prediction network is an LSTM, so nothing writes the KV cache.
    The attention shape below is a placeholder that keeps the per-token KV
    footprint tiny; the model never calls attention.

    The class keeps the transformers name because the Auto* mappings key on
    the config class name.
    """

    num_hidden_layers = 1
    num_attention_heads = 1
    num_key_value_heads = 1
    head_dim = 64
    # Bounds one request's prompt plus transcript; the transcription adapter
    # splits audio into clips short enough to stay far below it.
    max_position_embeddings = 8192

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.hidden_size = self.decoder_hidden_size
        # The model emits blank once the audio is exhausted, which ends the
        # request; it never emits blank otherwise.
        self.eos_token_id = self.blank_token_id
