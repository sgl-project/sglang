use futures::StreamExt;
use tokio::sync::mpsc;

use crate::{
    FrontendError, GenerationEvent, GenerationFinishReason, GenerationOutput, GenerationStream,
    MatchedStop,
};

use super::completions::SubmittedChoice;

fn submission() -> (
    GenerationStream,
    mpsc::Sender<Result<GenerationEvent, FrontendError>>,
) {
    let (tx, rx) = mpsc::channel(8);
    let events = futures::stream::unfold((rx, false), |(mut rx, finished)| async move {
        if finished {
            return None;
        }
        rx.recv().await.map(|item| {
            let finished = matches!(item, Ok(GenerationEvent::Done(_)) | Err(_));
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
    mpsc::Sender<Result<GenerationEvent, FrontendError>>,
) {
    let (events, tx) = submission();
    ((index, events), tx)
}

pub(super) fn submitted(
    index: usize,
    prompt_index: usize,
) -> (
    SubmittedChoice,
    mpsc::Sender<Result<GenerationEvent, FrontendError>>,
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

pub(super) fn chunk(text: &str, done: bool) -> Result<GenerationEvent, FrontendError> {
    let output = GenerationOutput {
        text: text.to_owned(),
        token_ids: vec![1],
        finish_reason: done
            .then(|| GenerationFinishReason::Stop(Some(MatchedStop::Text("</s>".into())))),
        prompt_tokens: 5,
        completion_tokens: 1,
        extras: None,
    };
    Ok(if done {
        GenerationEvent::Done(output)
    } else {
        GenerationEvent::Frame(output)
    })
}
