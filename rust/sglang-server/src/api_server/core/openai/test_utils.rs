//! Channel fixtures shared by the api_server::core openai test modules (and
//! re-exported to the api_server ones).

use crate::message::ids::Rid;
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::wiring::Senders;

pub(crate) fn senders() -> Senders {
    Senders {
        tok_manager_tx: flume::unbounded().0,
        abort_tx: flume::unbounded().0,
        tokenizer_tx: flume::unbounded().0,
        detokenizer_tx: vec![],
    }
}

pub(crate) fn chunk(rid: &str, text: &str, done: bool) -> ResponseItem {
    let output = ChunkEvent {
        rid: rid.into(),
        text: text.into(),
        token_ids: vec![1],
        prompt_tokens: 5,
        completion_tokens: 1,
        finish_reason: done.then(|| {
            serde_json::from_value(serde_json::json!({
                "type": "stop",
                "matched": "</s>"
            }))
            .unwrap()
        }),
        ..Default::default()
    };
    if done {
        ResponseItem::Done(output)
    } else {
        ResponseItem::Frame(output)
    }
}

/// A submitted chat choice (the tuple `chat_event_stream` consumes).
pub(crate) fn chat_submitted(
    index: usize,
    rid: &str,
) -> (
    (usize, Rid, tokio::sync::mpsc::Receiver<ResponseItem>),
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    ((index, rid.into(), rx), tx)
}

/// A submitted legacy completion choice.
pub(crate) fn submitted(
    index: usize,
    prompt_index: usize,
    rid: &str,
) -> (
    crate::api_server::core::openai::completions::SubmittedChoice,
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    (
        crate::api_server::core::openai::completions::SubmittedChoice {
            index,
            prompt_index,
            rid: rid.into(),
            echo: String::new(),
            rx,
        },
        tx,
    )
}
