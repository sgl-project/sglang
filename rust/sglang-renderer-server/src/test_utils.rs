use sglang_frontend::{PositionLogprobs, TokenLogprob};
use sglang_renderer::{RendererConfig, RendererLimits, SamplingDefaults};

fn logprob(token_id: i32, logprob: f32) -> TokenLogprob {
    TokenLogprob {
        logprob: Some(logprob),
        token_id,
        text: None,
    }
}

pub(crate) fn position(token_id: i32, value: f32, top: &[(i32, f32)]) -> PositionLogprobs {
    PositionLogprobs {
        token: logprob(token_id, value),
        top: top
            .iter()
            .map(|&(token_id, logprob)| self::logprob(token_id, logprob))
            .collect(),
    }
}

pub(crate) fn tiny_tokenizer() -> dynamo_tokenizers::Tokenizer {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experimental/sgl-router/tests/fixtures/tiny_tokenizer.json");
    dynamo_tokenizers::Tokenizer::from_file_with_options(
        path.to_str().unwrap(),
        dynamo_tokenizers::TokenizerOptions {
            add_special_tokens: false,
        },
    )
    .unwrap()
}

pub(crate) fn renderer_config() -> RendererConfig {
    RendererConfig {
        served_model_name: "model".into(),
        tokenizer_path: ".".into(),
        revision: None,
        model_path: String::new(),
        chat_template: Some("chatml".into()),
        tool_call_parser: None,
        reasoning_parser: None,
        default_chat_template_kwargs: Default::default(),
        stream_response_default_include_usage: false,
        default_sampling_params: SamplingDefaults::default(),
        limits: RendererLimits {
            vocab_size: 128,
            context_len: 128,
            num_reserved_tokens: 0,
            allow_auto_truncate: false,
            enable_return_hidden_states: false,
        },
    }
}
