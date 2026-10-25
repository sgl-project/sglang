//! External n-gram corpus management: tokenize on the frontend, load the same
//! chunks on every DP rank. Python's `TokenizerManager.add_external_corpus`
//! and friends.

use std::collections::BTreeMap;

use crate::message::ids::Rid;
use crate::message::io_struct::{
    AddExternalCorpusReqInput, ControlRequest, ListExternalCorporaReqInput,
    RemoveExternalCorpusReqInput,
};

use super::handle::CoreHandle;
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout;

/// Must match `SuffixAutomaton::kSeparatorToken` (and Python's
/// `external_corpus.SEPARATOR_TOKEN`): the document boundary inside a corpus.
const SEPARATOR_TOKEN: i64 = -(1 << 31);
/// Python's `DEFAULT_CHUNK_SIZE`: tokens per chunk streamed into the automaton.
const CHUNK_SIZE: usize = 4096;

/// What `/add_external_corpus` was given; exactly one source is used, in
/// this order of precedence, as in Python.
#[derive(Debug, Default)]
pub(crate) struct CorpusSource {
    pub(crate) corpus_id: Option<String>,
    pub(crate) file_path: Option<String>,
    pub(crate) documents: Option<Vec<String>>,
    pub(crate) token_chunks: Option<Vec<Vec<i64>>>,
}

#[derive(Debug, Default, PartialEq, serde::Deserialize)]
pub(crate) struct AddCorpusOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) corpus_id: String,
    #[serde(default)]
    pub(crate) message: String,
    #[serde(default)]
    pub(crate) loaded_token_count: i64,
}

#[derive(Debug, Default, PartialEq, serde::Deserialize)]
pub(crate) struct CorpusStatus {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

#[derive(Debug, Default, PartialEq, serde::Deserialize)]
pub(crate) struct CorporaListing {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) corpus_token_counts: BTreeMap<String, i64>,
    #[serde(default)]
    pub(crate) message: String,
}

const NGRAM_DISABLED: &str = "Ngram speculative decoding is not enabled.";

impl CoreHandle {
    fn ngram_enabled(&self) -> bool {
        self.inner
            .metadata
            .deployment_info
            .speculative_algorithm
            .as_deref()
            == Some("NGRAM")
    }

    /// Tokenize the corpus here and load it on every DP rank. Every failure
    /// is a `success: false` outcome with its reason, as Python answers.
    pub(crate) async fn add_external_corpus(&self, source: CorpusSource) -> AddCorpusOutcome {
        if !self.ngram_enabled() {
            return AddCorpusOutcome {
                message: NGRAM_DISABLED.into(),
                ..Default::default()
            };
        }
        match self.add_external_corpus_inner(source).await {
            Ok(outcome) => outcome,
            Err(error) => AddCorpusOutcome {
                message: error.to_string(),
                ..Default::default()
            },
        }
    }

    async fn add_external_corpus_inner(
        &self,
        source: CorpusSource,
    ) -> Result<AddCorpusOutcome, CoreError> {
        let max_tokens = self
            .inner
            .metadata
            .deployment_info
            .speculative_ngram_external_corpus_max_tokens;
        let corpus_id = source
            .corpus_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| Rid::new().to_string());
        let mut truncated = false;
        let token_chunks = if let Some(path) = source.file_path {
            self.chunks_from_file(&path, max_tokens).await?
        } else if let Some(documents) = source.documents {
            let (chunks, hit_limit) = self.chunks_from_documents(documents, max_tokens).await?;
            truncated = hit_limit;
            chunks
        } else if let Some(chunks) = source.token_chunks {
            chunks
        } else {
            return Ok(AddCorpusOutcome {
                message: "Either file_path or documents must be provided.".into(),
                ..Default::default()
            });
        };

        let outcomes = self
            .control(&|| {
                ControlRequest::AddExternalCorpusReqInput(AddExternalCorpusReqInput::new(
                    Rid::new().to_string(),
                    corpus_id.clone(),
                    token_chunks.clone(),
                ))
            })
            .await
            .decode::<AddCorpusOutcome>("add-external-corpus")?;
        let (success, mut message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        if truncated && success {
            message.push_str(&format!(" (truncated: exceeded {max_tokens} token limit)"));
        }
        let first = outcomes.into_iter().next().unwrap_or_default();
        Ok(AddCorpusOutcome {
            success,
            corpus_id: if success {
                first.corpus_id
            } else {
                String::new()
            },
            message,
            loaded_token_count: if success { first.loaded_token_count } else { 0 },
        })
    }

    /// Python's `iter_external_corpus_chunks`: a JSONL file of strings,
    /// documents separated by `SEPARATOR_TOKEN`, cut into `CHUNK_SIZE` pieces;
    /// exceeding `max_tokens` is an error.
    async fn chunks_from_file(
        &self,
        path: &str,
        max_tokens: u64,
    ) -> Result<Vec<Vec<i64>>, CoreError> {
        let invalid = |message: String| CoreError::InvalidArgument(message);
        if max_tokens == 0 {
            return Err(invalid(
                "External ngram corpus max tokens must be positive.".into(),
            ));
        }
        let contents = tokio::fs::read_to_string(path)
            .await
            .map_err(|_| invalid(format!("External ngram corpus path does not exist: {path}")))?;
        let mut chunks = Vec::new();
        let mut total_tokens: u64 = 0;
        let mut has_previous = false;
        for (index, line) in contents.lines().enumerate() {
            let line_no = index + 1;
            if line.trim().is_empty() {
                continue;
            }
            let record: serde_json::Value = serde_json::from_str(line).map_err(|error| {
                invalid(format!(
                    "Invalid JSON in external ngram corpus at line {line_no}: {error}"
                ))
            })?;
            let Some(text) = record.as_str() else {
                return Err(invalid(format!(
                    "Invalid external ngram corpus record at line {line_no}: expected a JSON string."
                )));
            };
            let mut token_ids = self.tokenize(text.to_owned(), false).await?;
            if token_ids.is_empty() {
                continue;
            }
            let separator_cost = u64::from(has_previous);
            let next_total = total_tokens + separator_cost + token_ids.len() as u64;
            if next_total > max_tokens {
                return Err(invalid(format!(
                    "External ngram corpus exceeds the configured token limit ({max_tokens}) \
                     at line {line_no} after loading {total_tokens} tokens."
                )));
            }
            total_tokens = next_total;
            if has_previous {
                token_ids.insert(0, SEPARATOR_TOKEN);
            }
            chunks.extend(token_ids.chunks(CHUNK_SIZE).map(<[i64]>::to_vec));
            has_previous = true;
        }
        Ok(chunks)
    }

    /// Python's inline `documents` path: one chunk per document, documents
    /// separated by `SEPARATOR_TOKEN`; stops (truncates) at `max_tokens`.
    async fn chunks_from_documents(
        &self,
        documents: Vec<String>,
        max_tokens: u64,
    ) -> Result<(Vec<Vec<i64>>, bool), CoreError> {
        let mut chunks = Vec::new();
        let mut total_tokens: u64 = 0;
        let mut has_previous = false;
        for document in documents {
            if document.is_empty() {
                continue;
            }
            let mut token_ids = self.tokenize(document, false).await?;
            if token_ids.is_empty() {
                continue;
            }
            if has_previous {
                token_ids.insert(0, SEPARATOR_TOKEN);
            }
            if total_tokens + token_ids.len() as u64 > max_tokens {
                return Ok((chunks, true));
            }
            total_tokens += token_ids.len() as u64;
            chunks.push(token_ids);
            has_previous = true;
        }
        Ok((chunks, false))
    }

    pub(crate) async fn remove_external_corpus(&self, corpus_id: String) -> CorpusStatus {
        if !self.ngram_enabled() {
            return CorpusStatus {
                success: false,
                message: NGRAM_DISABLED.into(),
            };
        }
        let outcomes = self
            .control(&|| {
                ControlRequest::RemoveExternalCorpusReqInput(RemoveExternalCorpusReqInput::new(
                    Rid::new().to_string(),
                    corpus_id.clone(),
                ))
            })
            .await
            .decode::<CorpusStatus>("remove-external-corpus");
        match outcomes {
            Ok(outcomes) => {
                let (success, message) = fanout::merge_status(
                    outcomes
                        .iter()
                        .map(|outcome| (outcome.success, outcome.message.as_str())),
                );
                CorpusStatus { success, message }
            }
            Err(error) => CorpusStatus {
                success: false,
                message: error.to_string(),
            },
        }
    }

    pub(crate) async fn list_external_corpora(&self) -> CorporaListing {
        if !self.ngram_enabled() {
            return CorporaListing {
                message: NGRAM_DISABLED.into(),
                ..Default::default()
            };
        }
        let outcomes = self
            .control(&|| {
                ControlRequest::ListExternalCorporaReqInput(ListExternalCorporaReqInput::new(
                    Rid::new().to_string(),
                ))
            })
            .await
            .decode::<CorporaListing>("list-external-corpora");
        match outcomes {
            Ok(mut outcomes) => {
                let (success, message) = fanout::merge_status(
                    outcomes
                        .iter()
                        .map(|outcome| (outcome.success, outcome.message.as_str())),
                );
                // Every rank loads the same set; the first rank's counts stand.
                let first = if outcomes.is_empty() {
                    CorporaListing::default()
                } else {
                    outcomes.swap_remove(0)
                };
                CorporaListing {
                    success,
                    corpus_token_counts: if success {
                        first.corpus_token_counts
                    } else {
                        BTreeMap::new()
                    },
                    message,
                }
            }
            Err(error) => CorporaListing {
                message: error.to_string(),
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::test_utils::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::config::DeploymentInfo;
    use crate::message::request::RequestKind;
    use crate::message::response::ResponseItem;
    use bytes::Bytes;

    fn ngram_handle(
        max_tokens: u64,
    ) -> (
        CoreHandle,
        flume::Receiver<crate::tokenizer_manager::wiring::TmEvent>,
    ) {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                metadata: CoreMetadata {
                    deployment_info: DeploymentInfo {
                        speculative_algorithm: Some("NGRAM".into()),
                        speculative_ngram_external_corpus_max_tokens: max_tokens,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: Vec::new(),
            },
        );
        (handle, intake_rx)
    }

    /// Documents are tokenized on the frontend (one chunk each, separated by
    /// the automaton's separator), the chunks ride the control request, and
    /// the scheduler's outcome comes back merged. Without NGRAM speculative
    /// decoding nothing reaches a scheduler.
    #[tokio::test]
    async fn add_external_corpus_tokenizes_documents_and_relays_chunks() {
        let harness = Harness::unbounded(8);
        let outcome = harness
            .handle
            .add_external_corpus(CorpusSource::default())
            .await;
        assert_eq!(
            outcome,
            AddCorpusOutcome {
                message: NGRAM_DISABLED.into(),
                ..Default::default()
            }
        );
        assert!(harness.intake_rx.try_recv().is_err());

        let (handle, intake_rx) = ngram_handle(100);
        let source = CorpusSource {
            documents: Some(vec!["one".into(), String::new(), "two".into()]),
            ..Default::default()
        };
        let (outcome, chunks) = tokio::join!(handle.add_external_corpus(source), async {
            let mut ids = vec![vec![1i64, 2], vec![3]].into_iter();
            loop {
                let request = accept_intake(intake_rx.recv_async().await.unwrap());
                match &request.kind {
                    RequestKind::Tokenize {
                        text,
                        add_special_tokens,
                    } => {
                        assert!(!add_special_tokens, "{text}: no BOS on corpus text");
                        request
                            .sink
                            .try_send(ResponseItem::TokenIds(ids.next().unwrap()))
                            .unwrap();
                    }
                    RequestKind::Control(control) => {
                        let rmpv::Value::Array(items) =
                            rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
                        else {
                            panic!("positional wire");
                        };
                        request
                            .sink
                            .try_send(ResponseItem::Control(Bytes::from(
                                rmp_serde::to_vec_named(&serde_json::json!({
                                    "success": true, "corpus_id": "c", "message": "",
                                    "loaded_token_count": 4,
                                }))
                                .unwrap(),
                            )))
                            .unwrap();
                        // [tag, rid, ipc, corpus_id, file_path, documents, token_chunks]
                        assert!(items[3].as_str().is_some_and(|id| !id.is_empty()));
                        assert!(items[4].is_nil() && items[5].is_nil());
                        return items[6].clone();
                    }
                    other => panic!("unexpected {other:?}"),
                }
            }
        });
        assert_eq!(
            chunks,
            rmpv::Value::Array(vec![
                rmpv::Value::Array(vec![1.into(), 2.into()]),
                rmpv::Value::Array(vec![SEPARATOR_TOKEN.into(), 3.into()]),
            ])
        );
        assert_eq!(
            outcome,
            AddCorpusOutcome {
                success: true,
                corpus_id: "c".into(),
                message: String::new(),
                loaded_token_count: 4,
            }
        );
    }

    /// Past the token limit the documents path truncates and says so; the
    /// file path is an error, as in Python.
    #[tokio::test]
    async fn corpus_token_limit_truncates_documents_and_rejects_files() {
        let (handle, intake_rx) = ngram_handle(2);
        let source = CorpusSource {
            documents: Some(vec!["a".into(), "b".into()]),
            ..Default::default()
        };
        let (outcome, ()) = tokio::join!(handle.add_external_corpus(source), async {
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            request
                .sink
                .try_send(ResponseItem::TokenIds(vec![7, 8]))
                .unwrap();
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            request
                .sink
                .try_send(ResponseItem::TokenIds(vec![9]))
                .unwrap();
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            assert!(matches!(request.kind, RequestKind::Control(_)));
            request
                .sink
                .try_send(ResponseItem::Control(Bytes::from(
                    rmp_serde::to_vec_named(
                        &serde_json::json!({"success": true, "corpus_id": "c"}),
                    )
                    .unwrap(),
                )))
                .unwrap();
        });
        assert!(outcome.success);
        assert_eq!(outcome.message, " (truncated: exceeded 2 token limit)");

        let dir = std::env::temp_dir().join(format!("corpus-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.jsonl");
        std::fs::write(&path, "\"hello\"\n\n\"world\"\n").unwrap();
        let source = CorpusSource {
            file_path: Some(path.to_string_lossy().into_owned()),
            ..Default::default()
        };
        let (outcome, ()) = tokio::join!(handle.add_external_corpus(source), async {
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            request
                .sink
                .try_send(ResponseItem::TokenIds(vec![1, 2]))
                .unwrap();
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            request
                .sink
                .try_send(ResponseItem::TokenIds(vec![3]))
                .unwrap();
        });
        assert!(!outcome.success);
        assert!(
            outcome
                .message
                .contains("exceeds the configured token limit (2) at line 3"),
            "{}",
            outcome.message
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
