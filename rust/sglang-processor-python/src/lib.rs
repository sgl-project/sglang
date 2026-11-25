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

    #[staticmethod]
    fn supports_reasoning_parser(name: &str) -> bool {
        sglang_processor::ReasoningStream::supports(name)
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
            self.encode_body(body)
        })
    }

    fn encode_reasoning_chat(
        &self,
        py: Python<'_>,
        request_json: &[u8],
        name: &str,
    ) -> PyResult<(Vec<u32>, ReasoningDecoder)> {
        py.detach(|| {
            let body: Value = serde_json::from_slice(request_json).map_err(invalid)?;
            let decoder = self.reasoning_decoder(name, &body)?;
            Ok((self.encode_body(body)?, decoder))
        })
    }
}

impl ChatProcessor {
    fn reasoning_decoder(&self, name: &str, body: &Value) -> PyResult<ReasoningDecoder> {
        let model = sglang_processor::openai::ChatModel::for_formatter(&self.formatter);
        let previous = body["messages"]
            .as_array()
            .and_then(|m| m.last())
            .filter(|m| body["continue_final_message"] == true && m["role"] == "assistant");
        let options = sglang_processor::ReasoningOptions {
            force_reasoning: model.request_thinking(Some(name), body, &self.default_kwargs),
            previous_content: previous
                .and_then(|m| m["content"].as_str())
                .map(str::to_owned),
            stream_reasoning: body["stream_reasoning"].as_bool().unwrap_or(true),
            force_nonempty_content: body["chat_template_kwargs"]["force_nonempty_content"] == true,
        };
        sglang_processor::ReasoningStream::new(name, &options)
            .map(ReasoningDecoder)
            .ok_or_else(|| invalid(format!("unsupported reasoning parser {name}")))
    }

    fn encode_body(&self, mut body: Value) -> PyResult<Vec<u32>> {
        let defaults = &self.default_kwargs;
        let render_generic = |body: Value| {
            let args = TemplateArgs::new(&body, defaults);
            let typed = serde_json::from_value(body).map_err(invalid)?;
            self.formatter
                .render_template(&typed, &args)
                .map_err(invalid)
        };
        if let Some(ids) = self.formatter.take_and_encode_request(&mut body, defaults) {
            return ids.map_err(invalid);
        }
        // Only formatters without a model-specific request renderer use the
        // generic template. A model-specific decline must reach the caller.
        let (prompt, continuation) = match &self.formatter {
            ChatFormatter::HuggingFace { .. } => {
                match self.formatter.render_text_jinja(&body, defaults) {
                    // Non-text callers retain the generic template's tool/media support.
                    Err(sglang_processor::TemplateError::NeedsRenderFallback { .. }) => {
                        render_generic(body)
                    }
                    rendered => rendered.map_err(invalid),
                }
            }
            ChatFormatter::Native { .. }
            | ChatFormatter::KimiK25 { .. }
            | ChatFormatter::Legacy(_) => render_generic(body),
            formatter => formatter.render_request(body, defaults).map_err(invalid),
        }?;
        let ids = if matches!(
            self.formatter,
            ChatFormatter::Legacy(_) | ChatFormatter::DeepSeekV32
        ) {
            // Both worker paths encode the prompt with tokenizer defaults.
            self.tokenizer
                .encode(prompt.as_str(), true)
                .and_then(|mut ids| {
                    self.tokenizer
                        .append_chat_continuation(&mut ids, &continuation)?;
                    Ok(ids)
                })
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
    }
}

#[pyclass(module = "sglang_processor._native")]
struct ReasoningDecoder(sglang_processor::ReasoningStream);

#[pymethods]
impl ReasoningDecoder {
    fn push(&mut self, text: &str, finished: bool) -> (String, String) {
        self.0.push(text, finished)
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
