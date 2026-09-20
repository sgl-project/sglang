use crate::{PositionLogprobs, TokenLogprob};

fn logprob(token_id: i32, logprob: f32) -> TokenLogprob {
    TokenLogprob {
        logprob: Some(logprob),
        token_id,
        text: None,
    }
}

pub(super) fn position(token_id: i32, value: f32, top: &[(i32, f32)]) -> PositionLogprobs {
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
