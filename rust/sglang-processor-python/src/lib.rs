use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use sglang_processor::dynamo_tokenizers::{DecodeStream, EncodeSegment};
use sglang_processor::{
    ChatFormatter, ChatFormatterOptions, DynamoTokenizer, OneOrMany, TemplateArgs, TextTokenizer,
    select_chat_formatter,
};

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(frozen, module = "sglang_processor._native")]
struct ChatProcessor {
    formatter: ChatFormatter,
    tokenizer: DynamoTokenizer,
    default_kwargs: HashMap<String, Value>,
}

#[pymethods]
impl ChatProcessor {
    #[new]
    #[pyo3(signature = (model_dir, chat_template=None, *, default_kwargs_json=None))]
    fn new(
        py: Python<'_>,
        model_dir: String,
        chat_template: Option<String>,
        default_kwargs_json: Option<&str>,
    ) -> PyResult<Self> {
        py.detach(|| {
            let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
                tokenizer_path: model_dir.clone(),
                model_path: model_dir.clone(),
                chat_template,
                ..Default::default()
            });
            Ok(Self {
                formatter: formatter.ok_or_else(|| {
                    invalid(error.unwrap_or_else(|| "missing chat template".into()))
                })?,
                default_kwargs: default_kwargs_json
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(invalid)?
                    .unwrap_or_default(),
                tokenizer: DynamoTokenizer::load(&model_dir, None).map_err(invalid)?,
            })
        })
    }

    #[getter]
    fn template_stops(&self) -> Vec<String> {
        match self.formatter.stop_strs() {
            Some(OneOrMany::One(stop)) => vec![stop],
            Some(OneOrMany::Many(stops)) => stops,
            None => Vec::new(),
        }
    }

    fn token_id(&self, token: &str) -> PyResult<Option<u32>> {
        let tokenizer = self.tokenizer.tokenizer();
        if let Ok(id) = tokenizer.token_to_id(token) {
            return Ok(id);
        }
        // Backends without vocabulary lookup can still encode a named special token.
        let ids = tokenizer
            .encode_segments(&[EncodeSegment::new(token, true)])
            .map_err(invalid)?;
        Ok(match ids.token_ids() {
            &[id] => Some(id),
            _ => None,
        })
    }

    fn decoder(&self, prompt_tail: Vec<u32>, skip_special_tokens: bool) -> Decoder {
        Decoder(
            self.tokenizer
                .tokenizer()
                .decode_stream(&prompt_tail, skip_special_tokens),
        )
    }

    /// Render and encode one serialized OpenAI chat request.
    fn encode_chat(&self, py: Python<'_>, request_json: &[u8]) -> PyResult<Vec<u32>> {
        py.detach(|| {
            let body: Value = serde_json::from_slice(request_json).map_err(invalid)?;
            let defaults = &self.default_kwargs;
            if let Some(ids) = self.formatter.encode_request(body.clone(), defaults) {
                return ids.map_err(invalid);
            }
            // Only formatters without a model-specific request renderer use the
            // generic template. A model-specific decline must reach the caller.
            let (prompt, continuation) = match &self.formatter {
                ChatFormatter::HuggingFace { .. }
                | ChatFormatter::KimiK25 { .. }
                | ChatFormatter::Legacy(_) => {
                    let args = TemplateArgs::new(&body, defaults);
                    let typed = serde_json::from_value(body).map_err(invalid)?;
                    self.formatter.render_template(&typed, &args)
                }
                formatter => formatter.render_request(body, defaults),
            }
            .map_err(invalid)?;
            let ids = if matches!(self.formatter, ChatFormatter::Legacy(_)) {
                // _apply_conversation_template encodes the entire prompt with tokenizer defaults.
                self.tokenizer.encode(prompt.as_str(), true)
            } else {
                let segments = prompt
                    .encode_segments()
                    .unwrap_or_else(|| vec![EncodeSegment::new(prompt.as_str(), true)]);
                self.tokenizer.encode_chat(&segments, &continuation)
            }
            .map_err(invalid)?;
            ids.into_iter()
                .map(|id| u32::try_from(id).map_err(invalid))
                .collect()
        })
    }
}

/// One incremental decoder per generation. Token boundaries can split UTF-8.
#[pyclass(module = "sglang_processor._native")]
struct Decoder(DecodeStream);

#[pymethods]
impl Decoder {
    fn push(&mut self, ids: Vec<u32>, finished: bool) -> PyResult<String> {
        let mut text = String::new();
        for id in ids {
            if let Some(piece) = self.0.step(id).map_err(invalid)? {
                text.push_str(&piece);
            }
        }
        if finished && let Some(tail) = self.0.finish().map_err(invalid)? {
            text.push_str(&tail);
        }
        Ok(text)
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ChatProcessor>()
}
