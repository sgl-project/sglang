//! Prompt and generated-token decoding, including local text stops.

use super::{internal, invalid};
use crate::{
    GenerateRequest, GenerationOutput, GenerationOutputExtras, ResponseError, TokenIds,
    TokenLogprob,
};

use super::{GenerationFinishReason, GenerationStream, MatchedStop, TokenStream};
use futures::StreamExt;

/// Shared tokenizer handle for prompt and generated-output decoding.
pub(crate) struct TokenDecoder {
    tokenizer: dynamo_tokenizers::Tokenizer,
}

pub(super) struct DecodeState {
    decoder: dynamo_tokenizers::DecodeStream,
    stops: Option<StopStringMatcher>,
    logprob_text: bool,
}

impl TokenDecoder {
    pub(crate) fn new(tokenizer: dynamo_tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub(crate) fn detokenize_prompt(&self, token_ids: TokenIds) -> Result<String, ResponseError> {
        let ids = token_ids
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("token IDs must be non-negative"))?;
        self.tokenizer
            .decode(&ids, true)
            .map(String::from)
            .map_err(|error| invalid(format!("detokenizing prompt failed: {error}")))
    }

    pub(super) fn prepare(
        &self,
        request: &mut GenerateRequest,
    ) -> Result<DecodeState, ResponseError> {
        let stops = take_text_stops(request)?;
        let prompt_ids = request
            .input_ids
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("input_ids must be non-negative"))?;
        let logprob_text = request.return_text_in_logprobs.unwrap_or(false);
        request.return_text_in_logprobs = Some(false);
        Ok(DecodeState {
            decoder: self
                .tokenizer
                .decode_stream(&prompt_ids, request.sampling_params.skip_special_tokens),
            stops,
            logprob_text,
        })
    }

    pub(super) fn decode(
        &self,
        mut tokens: TokenStream,
        mut state: DecodeState,
    ) -> GenerationStream {
        let tokenizer = self.tokenizer.clone();
        async_stream::try_stream! {
            while let Some(delta) = tokens.next().await {
                let mut output = GenerationOutput::from(delta?);
                let matched = decode_output(&mut state.decoder, &mut output, state.stops.as_mut())?;
                if state.logprob_text {
                    fill_logprob_text(&tokenizer, output.extras.as_deref_mut());
                }
                let stopped = matched.is_some();
                if let Some(stop) = matched {
                    output.finish_reason = Some(GenerationFinishReason::Stop(Some(MatchedStop::Text(stop))));
                }
                if stopped {
                    drop(tokens);
                    yield output;
                    return;
                }
                yield output;
            }
        }.boxed()
    }
}

pub(super) fn take_text_stops(
    request: &mut GenerateRequest,
) -> Result<Option<StopStringMatcher>, ResponseError> {
    let params = &mut request.sampling_params;
    let matcher = StopStringMatcher::new(std::mem::take(&mut params.stop), params.no_stop_trim);
    Ok(matcher)
}

pub(super) struct StopStringMatcher {
    stops: Vec<String>,
    pending: String,
    include_stop: bool,
}

struct StopMatch {
    text: String,
    matched: Option<String>,
}

impl StopStringMatcher {
    fn new(stops: Vec<String>, include_stop: bool) -> Option<Self> {
        (!stops.is_empty()).then_some(Self {
            stops,
            pending: String::new(),
            include_stop,
        })
    }

    fn push(&mut self, text: &str) -> StopMatch {
        self.pending.push_str(text);
        if let Some((position, stop)) = self
            .stops
            .iter()
            .filter_map(|stop| {
                self.pending
                    .find(stop)
                    .map(|position| (position, stop.clone()))
            })
            .min_by_key(|(position, _)| *position)
        {
            if stop.is_empty() {
                return StopMatch {
                    text: std::mem::take(&mut self.pending),
                    matched: Some(stop),
                };
            }
            let end = if self.include_stop {
                position + stop.len()
            } else {
                position
            };
            let text = self.pending[..end].to_owned();
            self.pending.clear();
            return StopMatch {
                text,
                matched: Some(stop),
            };
        }

        let held_start = self
            .pending
            .char_indices()
            .map(|(start, _)| start)
            .chain(std::iter::once(self.pending.len()))
            .find(|&start| {
                self.stops
                    .iter()
                    .any(|stop| stop.starts_with(&self.pending[start..]))
            })
            .unwrap_or(self.pending.len());
        let held = self.pending.split_off(held_start);
        let text = std::mem::replace(&mut self.pending, held);
        StopMatch {
            text,
            matched: None,
        }
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

pub(super) fn decode_output(
    decoder: &mut dynamo_tokenizers::DecodeStream,
    output: &mut GenerationOutput,
    mut stop_matcher: Option<&mut StopStringMatcher>,
) -> Result<Option<String>, ResponseError> {
    let mut text = String::new();
    for index in 0..output.token_ids.len() {
        let id = output.token_ids[index];
        let id = u32::try_from(id).map_err(|_| internal("engine returned a negative token ID"))?;
        let delta = decoder
            .step(id)
            .map_err(|error| internal(format!("detokenizing engine output failed: {error}")))?;
        if let Some(matcher) = stop_matcher.as_deref_mut() {
            let matched = matcher.push(delta.as_deref().unwrap_or_default());
            text.push_str(&matched.text);
            if let Some(stop) = matched.matched {
                truncate_output(output, index + 1)?;
                output.text = text;
                return Ok(Some(stop));
            }
        } else if let Some(delta) = delta {
            text.push_str(&delta);
        }
    }
    if output.finish_reason.is_some()
        && let Some(matcher) = stop_matcher
    {
        text.push_str(&matcher.flush());
    }
    output.text = text;
    Ok(None)
}

fn truncate_output(output: &mut GenerationOutput, kept_tokens: usize) -> Result<(), ResponseError> {
    output.token_ids.truncate(kept_tokens);
    output.completion_tokens = u64::try_from(kept_tokens).unwrap_or(u64::MAX);
    let Some(extras) = output.extras.as_deref_mut() else {
        return Ok(());
    };
    truncate_optional(
        &mut extras.output_logprobs,
        kept_tokens,
        "output logprob positions",
    )
}

fn truncate_optional<T>(
    values: &mut Vec<T>,
    length: usize,
    description: &str,
) -> Result<(), ResponseError> {
    if values.is_empty() {
        return Ok(());
    }
    if values.len() < length {
        return Err(internal(format!(
            "engine returned {} {description} values for {length} retained tokens",
            values.len()
        )));
    }
    values.truncate(length);
    Ok(())
}

pub(super) fn fill_logprob_text(
    tokenizer: &dynamo_tokenizers::Tokenizer,
    extras: Option<&mut GenerationOutputExtras>,
) {
    let Some(extras) = extras else { return };
    for position in extras
        .output_logprobs
        .iter_mut()
        .chain(&mut extras.input_logprobs)
    {
        fill_text(tokenizer, &mut position.token);
        for token in &mut position.top {
            fill_text(tokenizer, token);
        }
    }
}

fn fill_text(tokenizer: &dynamo_tokenizers::Tokenizer, token: &mut TokenLogprob) {
    if token.text.is_some() {
        return;
    }
    token.text = Some(
        u32::try_from(token.token_id)
            .ok()
            .and_then(|id| tokenizer.decode(&[id], false).ok())
            .map(String::from)
            .unwrap_or_default(),
    );
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{position, tiny_tokenizer};
    use super::*;
    use crate::{GenerationOptions, SamplingParams, TokenIdsRequest};

    fn request(stop: Vec<&str>) -> GenerateRequest {
        TokenIdsRequest {
            rid: "r".into(),
            input_ids: vec![1],
            options: GenerationOptions {
                sampling_params: SamplingParams {
                    stop_strs: stop.into_iter().map(str::to_owned).collect(),
                    ..Default::default()
                },
                ..Default::default()
            },
            metadata: Default::default(),
        }
        .into()
    }

    #[test]
    fn text_stops_stay_in_the_frontend_and_token_stops_reach_the_engine() {
        let mut request = request(vec!["<eos>"]);
        request.sampling_params.stop_token_ids = Some(vec![9]);
        let matcher = take_text_stops(&mut request).unwrap();

        assert!(matcher.is_some());
        assert_eq!(request.sampling_params.stop_token_ids, Some(vec![9]));
        assert!(request.sampling_params.stop.is_empty());
    }

    #[test]
    fn regex_stops_and_min_tokens_reach_the_engine() {
        let mut request = request(vec!["END"]);
        request.sampling_params.stop_regex = vec!["[0-9]{3}".into()];
        request.sampling_params.min_new_tokens = 4;

        take_text_stops(&mut request).unwrap();

        assert_eq!(request.sampling_params.stop_regex, ["[0-9]{3}"]);
        assert_eq!(request.sampling_params.min_new_tokens, 4);
    }

    #[test]
    fn decoded_stop_matcher_handles_cross_frame_matches_and_order() {
        let mut matcher = StopStringMatcher::new(vec!["END".into(), "ND".into()], false).unwrap();

        let first = matcher.push("value E");
        assert_eq!(first.text, "value ");
        assert!(first.matched.is_none());

        let second = matcher.push("ND trailing");
        assert_eq!(second.text, "");
        assert_eq!(second.matched.as_deref(), Some("END"));
    }

    #[test]
    fn decoded_stop_matcher_uses_the_earliest_match() {
        let mut matcher =
            StopStringMatcher::new(vec!["later".into(), "first".into()], false).unwrap();

        let matched = matcher.push("first then later");

        assert_eq!(matched.text, "");
        assert_eq!(matched.matched.as_deref(), Some("first"));
    }

    #[test]
    fn no_stop_trim_includes_the_matched_text() {
        let mut matcher = StopStringMatcher::new(vec!["END".into()], true).unwrap();
        let matched = matcher.push("value END trailing");

        assert_eq!(matched.text, "value END");
        assert_eq!(matched.matched.as_deref(), Some("END"));
    }

    #[test]
    fn local_stop_truncates_token_aligned_logprobs() {
        let mut output = GenerationOutput {
            token_ids: vec![7, 8, 9],
            completion_tokens: 3,
            extras: Some(Box::new(GenerationOutputExtras {
                output_logprobs: vec![
                    position(7, -0.1, &[(7, -0.1), (6, -1.0)]),
                    position(8, -0.2, &[(8, -0.2)]),
                    position(9, -0.3, &[(9, -0.3)]),
                ],
                ..Default::default()
            })),
            ..Default::default()
        };

        truncate_output(&mut output, 2).unwrap();

        assert_eq!(output.token_ids, [7, 8]);
        assert_eq!(output.completion_tokens, 2);
        let extras = output.extras.unwrap();
        assert_eq!(extras.output_logprobs.len(), 2);
        assert_eq!(extras.output_logprobs[0].top.len(), 2);
        assert_eq!(extras.output_logprobs[1].top.len(), 1);
        assert_eq!(extras.output_logprobs[1].token.token_id, 8);
    }

    #[test]
    fn text_stops_are_matched_on_contextual_decoder_output() {
        let tokenizer = tiny_tokenizer();
        let token_ids = tokenizer
            .encode("hello")
            .unwrap()
            .token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let mut expected_decoder = tokenizer.decode_stream(&[65], true);
        let mut decoded = String::new();
        for &id in &token_ids {
            if let Some(delta) = expected_decoder.step(id as u32).unwrap() {
                decoded.push_str(&delta);
            }
        }
        assert!(!decoded.is_empty());

        let mut decoder = tokenizer.decode_stream(&[65], true);
        let mut output = GenerationOutput {
            token_ids,
            completion_tokens: 1,
            ..Default::default()
        };
        let mut matcher = StopStringMatcher::new(vec![decoded.clone()], false).unwrap();

        let matched = decode_output(&mut decoder, &mut output, Some(&mut matcher)).unwrap();

        assert_eq!(matched.as_deref(), Some(decoded.as_str()));
        assert!(output.text.is_empty());
    }

    #[test]
    fn empty_stop_matches_after_the_first_generated_token() {
        let tokenizer = tiny_tokenizer();
        let mut decoder = tokenizer.decode_stream(&[65], true);
        let mut output = GenerationOutput {
            token_ids: vec![104, 101],
            completion_tokens: 2,
            ..Default::default()
        };
        let mut matcher = StopStringMatcher::new(vec!["never".into(), String::new()], false)
            .expect("the empty stop must remain active");

        let matched = decode_output(&mut decoder, &mut output, Some(&mut matcher)).unwrap();

        assert_eq!(matched.as_deref(), Some(""));
        assert_eq!(output.token_ids, [104]);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(output.text, "h");
    }
}
