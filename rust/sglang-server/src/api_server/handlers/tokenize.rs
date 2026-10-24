//! Tokenizer-pool and detokenizer-shard unary calls.

use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::message::types::TokenIds;
use crate::utils::error::Error;

use super::handle::{CoreHandle, translate_runtime_error};
use crate::api_server::core::CoreError;

impl CoreHandle {
    /// Decode token IDs into text without exposing the runtime's byte payload.
    pub(crate) async fn detokenize(
        &self,
        token_ids: TokenIds,
        skip_special_tokens: bool,
    ) -> Result<String, CoreError> {
        let mut call = self
            .submit(RequestKind::Detokenize {
                token_ids,
                skip_special_tokens,
            })
            .await?;
        let bytes = match call.recv().await {
            Some(ResponseItem::Data(bytes)) => bytes,
            Some(ResponseItem::Error(Error::Validation(message))) => {
                return Err(CoreError::InvalidArgument(message));
            }
            Some(ResponseItem::Error(error)) => return Err(translate_runtime_error(error)),
            Some(_) => {
                return Err(CoreError::InvalidResponse(
                    "unexpected output for detokenize request".into(),
                ));
            }
            None => {
                return Err(CoreError::Internal("reply channel closed".into()));
            }
        };
        String::from_utf8(bytes.to_vec())
            .map_err(|_| CoreError::InvalidResponse("detokenized prompt is not valid UTF-8".into()))
    }

    /// Encode one text into token IDs on the tokenizer pool.
    pub(crate) async fn tokenize(
        &self,
        text: String,
        add_special_tokens: bool,
    ) -> Result<TokenIds, CoreError> {
        let mut call = self
            .submit(RequestKind::Tokenize {
                text,
                add_special_tokens,
            })
            .await?;
        match call.recv().await {
            Some(ResponseItem::TokenIds(ids)) => Ok(ids),
            Some(ResponseItem::Error(Error::Validation(message))) => {
                Err(CoreError::InvalidArgument(message))
            }
            Some(ResponseItem::Error(error)) => Err(translate_runtime_error(error)),
            Some(_) => Err(CoreError::InvalidResponse(
                "unexpected output for tokenize request".into(),
            )),
            None => Err(CoreError::Internal("reply channel closed".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::test_utils::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn token_operations_mint_internal_rids() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let task = tokio::spawn(async move {
            handle
                .detokenize(vec![1, 2], false /* skip_special_tokens */)
                .await
        });

        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(!request.rid.as_str().is_empty());
        assert!(matches!(
            request.kind,
            RequestKind::Detokenize {
                skip_special_tokens: false,
                ..
            }
        ));
        request
            .sink
            .try_send(ResponseItem::Data(Bytes::from_static(b"decoded")))
            .unwrap();
        assert_eq!(task.await.unwrap().unwrap(), "decoded");

        let handle = harness.handle.clone();
        let task = tokio::spawn(async move {
            handle
                .tokenize("hi".into(), true /* add_special_tokens */)
                .await
        });
        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(!request.rid.as_str().is_empty());
        assert!(matches!(
            &request.kind,
            RequestKind::Tokenize {
                text,
                add_special_tokens: true,
            } if text == "hi"
        ));
        request
            .sink
            .try_send(ResponseItem::TokenIds(vec![1, 2]))
            .unwrap();
        assert_eq!(task.await.unwrap().unwrap(), vec![1, 2]);
        assert!(harness.abort_rx.try_recv().is_err());
    }
}
