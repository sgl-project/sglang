use futures::StreamExt;
use futures::future::BoxFuture;
use tokio::sync::mpsc;

use crate::{
    FrontendError, GenerationEvent, GenerationFinishReason, GenerationInput, GenerationOutput,
    GenerationSubmission, InferenceSession, MatchedStop, TokenIds,
};

use super::completions::SubmittedChoice;

pub(super) struct TestSession;

impl InferenceSession for TestSession {
    fn submit(
        &mut self,
        _request: GenerationInput,
        _stream: bool,
    ) -> BoxFuture<'_, Result<GenerationSubmission, FrontendError>> {
        Box::pin(async { unreachable!("tests inject submissions directly") })
    }

    fn detokenize(&mut self, _token_ids: TokenIds) -> BoxFuture<'_, Result<String, FrontendError>> {
        Box::pin(async { unreachable!("tests do not detokenize") })
    }

    fn complete(&mut self, _submission_id: &str) {}
}

fn submission(
    id: &str,
) -> (
    GenerationSubmission,
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
    (
        GenerationSubmission {
            id: id.to_owned(),
            events,
        },
        tx,
    )
}

pub(super) fn chat_submitted(
    index: usize,
    id: &str,
) -> (
    (usize, GenerationSubmission),
    mpsc::Sender<Result<GenerationEvent, FrontendError>>,
) {
    let (submission, tx) = submission(id);
    ((index, submission), tx)
}

pub(super) fn submitted(
    index: usize,
    prompt_index: usize,
    id: &str,
) -> (
    SubmittedChoice,
    mpsc::Sender<Result<GenerationEvent, FrontendError>>,
) {
    let (submission, tx) = submission(id);
    (
        SubmittedChoice {
            index,
            prompt_index,
            echo: String::new(),
            submission,
        },
        tx,
    )
}

pub(super) fn chunk(_id: &str, text: &str, done: bool) -> Result<GenerationEvent, FrontendError> {
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
