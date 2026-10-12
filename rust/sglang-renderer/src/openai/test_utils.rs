use crate::{RendererConfig, RendererLimits, SamplingDefaults};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::{
    GenerationFinishReason, GenerationOutput, GenerationStream, MatchedStop, ResponseError,
};

use super::completions::SubmittedChoice;

fn submission() -> (
    GenerationStream,
    mpsc::Sender<Result<GenerationOutput, ResponseError>>,
) {
    let (tx, rx) = mpsc::channel::<Result<GenerationOutput, ResponseError>>(8);
    let events = futures::stream::unfold((rx, false), |(mut rx, finished)| async move {
        if finished {
            return None;
        }
        rx.recv().await.map(|item| {
            let finished = match &item {
                Ok(output) => output.finish_reason.is_some(),
                Err(_) => true,
            };
            (item, (rx, finished))
        })
    })
    .boxed();
    (events, tx)
}

pub(super) fn chat_submitted(
    index: usize,
) -> (
    (usize, GenerationStream),
    mpsc::Sender<Result<GenerationOutput, ResponseError>>,
) {
    let (events, tx) = submission();
    ((index, events), tx)
}

pub(super) fn submitted(
    index: usize,
    prompt_index: usize,
) -> (
    SubmittedChoice,
    mpsc::Sender<Result<GenerationOutput, ResponseError>>,
) {
    let (events, tx) = submission();
    (
        SubmittedChoice {
            index,
            prompt_index,
            echo: String::new(),
            events,
        },
        tx,
    )
}

pub(super) fn chunk(text: &str, done: bool) -> Result<GenerationOutput, ResponseError> {
    let output = GenerationOutput {
        text: text.to_owned(),
        token_ids: vec![1],
        finish_reason: done
            .then(|| GenerationFinishReason::Stop(Some(MatchedStop::Text("</s>".into())))),
        prompt_tokens: 5,
        completion_tokens: 1,
        extras: None,
    };
    Ok(output)
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
