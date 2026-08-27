//! SGLang engine output to renderer chat-event integration.
//!
//! This is the only layer that knows both scheduler [`ChunkEvent`] values and
//! renderer [`DecodedChatEvent`] values. Axum and future Tonic adapters consume
//! semantic [`ChatEvent`](sglang_renderer::ChatEvent) output instead.

use dynamo_protocols::types::{ChatChoiceLogprobs, ChatCompletionTokenLogprob, TopLogprobs};
use futures::StreamExt;
use sglang_renderer::{
    ChatFinishReason, ChatResponseError, ChatResponseInput, ChatResponseItem,
    ChatResponseProcessor, DecodedChatEvent,
};
use tokio::sync::mpsc;

use crate::frontend::AbortGuard;
use crate::message::ids::Rid;
use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};

pub(crate) fn semantic_chat_stream(
    submitted: Vec<(usize, Rid, mpsc::Receiver<ResponseItem>)>,
    mut guard: AbortGuard,
    response_processor: ChatResponseProcessor,
    want_logprobs: bool,
) -> impl futures::Stream<Item = ChatResponseItem> {
    let count = submitted.len();
    let raw = async_stream::stream! {
        let mut rids = Vec::with_capacity(count);
        let mut streams = Vec::with_capacity(count);

        for (index, rid, rx) in submitted {
            rids.push(rid);
            streams.push(indexed_decode_stream(index, rx));
        }

        let mut events = futures::stream::select_all(streams);
        while let Some((index, item)) = events.next().await {
            let Some(item) = item else {
                yield ChatResponseInput::Error(ChatResponseError {
                    status_code: 500,
                    message: "response truncated before completion".into(),
                });
                continue;
            };
            let output = match item {
                ResponseItem::Frame(output) => output,
                ResponseItem::Done(output) => {
                    guard.disarm(&rids[index]);
                    output
                }
                ResponseItem::Error(error) => {
                    guard.disarm(&rids[index]);
                    yield ChatResponseInput::Error(ChatResponseError {
                        status_code: error.http_status(),
                        message: error.to_string(),
                    });
                    continue;
                }
                ResponseItem::Control(_) | ResponseItem::Data(_) => continue,
            };
            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield ChatResponseInput::Error(ChatResponseError {
                    status_code: code,
                    message: message.to_owned(),
                });
                continue;
            }

            let finish_reason = chat_finish_reason(&output);
            yield ChatResponseInput::Decoded(DecodedChatEvent {
                choice: index,
                text: output.text,
                token_ids: output.token_ids,
                finish_reason,
                logprobs: want_logprobs.then(|| chat_logprobs(output.extras.as_deref())),
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            });
        }
    };

    response_processor.process_stream(raw)
}

pub(crate) fn chat_finish_reason(output: &ChunkEvent) -> Option<ChatFinishReason> {
    let kind = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.kind_name());
    kind.map(|kind| match kind {
        "length" => ChatFinishReason::Length,
        "content_filter" => ChatFinishReason::ContentFilter,
        _ => ChatFinishReason::Stop,
    })
}

#[allow(deprecated)]
pub(crate) fn chat_logprobs(extras: Option<&ChunkExtras>) -> ChatChoiceLogprobs {
    let mut content = Vec::new();
    let Some(extras) = extras else {
        return ChatChoiceLogprobs {
            content: Some(content),
            refusal: None,
        };
    };
    let mut top_offset = 0usize;
    for (position, (&logprob, &token_id)) in
        extras.out_lp_val.iter().zip(&extras.out_lp_idx).enumerate()
    {
        let token = extras
            .out_lp_txt
            .get(position)
            .cloned()
            .unwrap_or_else(|| format!("token_id:{token_id}"));
        let top_len = extras.out_top_lens.get(position).copied().unwrap_or(0) as usize;
        let top_logprobs = extras.out_top_val[top_offset..]
            .iter()
            .zip(&extras.out_top_idx[top_offset..])
            .take(top_len)
            .enumerate()
            .map(|(offset, (&logprob, &id))| {
                let text = extras
                    .out_top_txt
                    .get(top_offset + offset)
                    .cloned()
                    .unwrap_or_else(|| format!("token_id:{id}"));
                TopLogprobs {
                    bytes: Some(text.as_bytes().to_vec()),
                    token: text,
                    logprob,
                }
            })
            .collect();
        top_offset = top_offset.saturating_add(top_len);
        content.push(ChatCompletionTokenLogprob {
            bytes: Some(token.as_bytes().to_vec()),
            token,
            logprob,
            token_id: u32::try_from(token_id).ok(),
            top_logprobs,
        });
    }
    ChatChoiceLogprobs {
        content: Some(content),
        refusal: None,
    }
}

fn indexed_decode_stream(
    index: usize,
    rx: mpsc::Receiver<ResponseItem>,
) -> futures::stream::BoxStream<'static, (usize, Option<ResponseItem>)> {
    futures::stream::unfold((rx, false), move |(mut rx, finished)| async move {
        if finished {
            return None;
        }
        match rx.recv().await {
            Some(item) => {
                let finished = matches!(item, ResponseItem::Done(_) | ResponseItem::Error(_));
                Some(((index, Some(item)), (rx, finished)))
            }
            None => Some(((index, None), (rx, true))),
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_maps_scheduler_kinds() {
        let output = |finish: serde_json::Value| ChunkEvent {
            rid: "r".into(),
            text: "x".into(),
            token_ids: vec![1],
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: Some(serde_json::from_value(finish).unwrap()),
            ..Default::default()
        };
        assert_eq!(
            chat_finish_reason(&output(
                serde_json::json!({"type": "stop", "matched": "</s>"})
            )),
            Some(ChatFinishReason::Stop)
        );
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "length", "length": 8}))),
            Some(ChatFinishReason::Length)
        );
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "content_filter"}))),
            Some(ChatFinishReason::ContentFilter)
        );
    }
}
