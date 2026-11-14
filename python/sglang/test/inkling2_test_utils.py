"""Shared helpers for the model package tests."""


class ByteTokenizer:
    def __init__(self):
        names = [
            "message_user",
            "message_system",
            "message_model",
            "message_tool",
            "content_text",
            "content_thinking",
            "content_tool_error",
            "content_invoke_tool_json",
            "content_invoke_tool_text",
            "content_xml",
            "content_image",
            "end_message",
            "content_model_end_sampling",
        ]
        self.special_tokens = {name: index + 256 for index, name in enumerate(names)}
        self.names = {value: key for key, value in self.special_tokens.items()}

    def encode_special(self, name):
        return self.special_tokens[name]

    def decode_special(self, token):
        return self.names.get(token)

    def encode_ordinary(self, text):
        return list(text.encode())

    def decode_bytes(self, tokens):
        return b"".join(
            f"<|{self.names[token]}|>".encode()
            if token in self.names
            else bytes([token])
            for token in tokens
        )

    def decode_strict(self, tokens):
        return self.decode_bytes(tokens).decode()


def precision_group(compute, names):
    if compute == "nvfp4":
        return dict(
            tensors=names,
            output_dtype="uint8",
            compute_dtype="nvfp4",
            weight_transform="nvfp4_nearest",
            weight_block_shape=[1, 16],
            weight_scale_format="amax_denominator",
            required_activation_quantization=dict(
                format="nvfp4_e2m1",
                block_size=16,
                scale_dtype="float8_e4m3fn",
                rounding="nearest",
                amax_type="per_token",
                scale_selection_rule="fouroversix_mae",
            ),
        )
    quantized = compute == "mxfp8"
    return dict(
        tensors=names,
        output_dtype="float8_e4m3fn" if quantized else compute,
        compute_dtype=compute,
        weight_transform="mxfp8_nearest" if quantized else "cast",
        weight_block_shape=[32, 32] if quantized else [],
        required_activation_quantization=dict(
            format="mxfp8_e4m3", block_size=32, scale_dtype="ue8m0", rounding="nearest"
        )
        if quantized
        else None,
    )
