//! Request validation for scheduler intake.

use crate::message::config::HiddenStatesMode;
use crate::message::request::{GenerateRequest, Request, RequestKind};
use crate::utils::{
    error::Error,
    fsm::{Event, ValidationOutcome},
};

use super::to_scheduler::MAX_RID_LEN;
use super::to_scheduler_types::Limits;

/// `Received → Validating` + admissibility check. Under `skip_tokenizer_init` a
/// generate request must already carry token ids (no tokenizer to byte-encode
/// text); control requests carry none and are exempt.
pub(super) fn validate(req: &mut Request, limits: &Limits) -> Result<(), Error> {
    let (skip_tokenizer_init, vocab_size) = (limits.skip_tokenizer_init, limits.vocab_size);
    let _ = req
        .state
        .apply(Event::Validated(ValidationOutcome::NeedsTokenize));

    // The rid is the request's identity everywhere downstream: it keys the detok
    // table, and it rides on EVERY chunk of EVERY decode step. An unbounded
    // client-supplied rid is therefore a per-step cost, not a one-off.
    // Measured on the CLIENT-facing form: the uniquifier `Rid::from_client` appends
    // is this server's own overhead, and charging the client for bytes it did not
    // send would reject a rid exactly at the documented limit.
    let client_rid_len = req.rid.client_facing().len();
    if client_rid_len > MAX_RID_LEN {
        return Err(Error::Validation(format!(
            "rid is {client_rid_len} bytes, over the {MAX_RID_LEN}-byte limit"
        )));
    }
    if skip_tokenizer_init
        && matches!(&req.kind, RequestKind::Generate(g) if !g.already_tokenized())
    {
        // `Validation` (400), not `Tokenize` (500): the client sent a request this
        // server cannot serve, which is their error to fix — Python 400s it too.
        return Err(Error::Validation(
            "skip_tokenizer_init is set: request must provide input_ids".into(),
        ));
    }
    // No tokenizer pool runs under `skip_tokenizer_init` to answer it.
    if skip_tokenizer_init && matches!(&req.kind, RequestKind::Tokenize { .. }) {
        return Err(Error::Validation(
            "tokenization is unavailable when skip_tokenizer_init=True".into(),
        ));
    }

    // Multimodal processors may consume sentinel ids outside the vocabulary.
    // Validate their resulting ids at PreSendValidating instead. Non-MM client
    // ids can be rejected now.
    if let RequestKind::Generate(g) = &req.kind {
        if !g.has_multimodal() {
            validate_input_ids(g, vocab_size)?;
        }
        if let Some(ids) = &g.token_ids_logprob {
            for &id in ids {
                if id < 0 || id as u64 >= vocab_size {
                    return Err(Error::Validation(format!(
                        "token_ids_logprob contains out-of-vocabulary token id \
                         {id}; valid range is [0, {vocab_size})"
                    )));
                }
            }
        }
    }

    // Detokenize ids must fit the shard's `&[u32]` decode domain. No vocab
    // bound — parity with the retired direct decode service: an unknown id is
    // the tokenizer's error to report, and nothing here reaches the scheduler's
    // embedding lookup.
    if let RequestKind::Detokenize { token_ids, .. } = &req.kind {
        for &id in token_ids {
            if u32::try_from(id).is_err() {
                return Err(Error::Validation(format!("Token ID {id} is out of range")));
            }
        }
    }

    // Python `TokenizerManager._validate_one_request`.
    // A request above the server's mode raises in a CUDA graph captured for the lower mode.
    let requested = match &req.kind {
        RequestKind::Generate(g) if g.return_hidden_states => HiddenStatesMode::Full,
        _ => HiddenStatesMode::Null,
    };
    if requested > limits.return_hidden_states_mode {
        return Err(Error::Validation(
            match limits.return_hidden_states_mode {
                HiddenStatesMode::Null => {
                    "The server is not configured to return hidden states. \
                     Please set `--return-hidden-states-mode last`, \
                     `--return-hidden-states-mode full`, or the legacy \
                     `--enable-return-hidden-states` flag."
                }
                // Nothing is above `full`, so the exceeded maximum is `last`.
                HiddenStatesMode::Last | HiddenStatesMode::Full => {
                    "The requested return_hidden_states mode exceeds the server \
                     maximum `last`. Please launch with \
                     `--return-hidden-states-mode full` to allow \
                     return_hidden_states=True."
                }
            }
            .into(),
        ));
    }

    Ok(())
}

/// Guard the ids that will reach the embedding lookup. Multimodal requests run
/// this after placeholder expansion; all other requests also run it at intake.
pub(super) fn validate_input_ids(g: &GenerateRequest, vocab_size: u64) -> Result<(), Error> {
    if let Some(ids) = &g.input_ids {
        for &id in ids {
            if id < 0 || id as u64 >= vocab_size {
                return Err(Error::Validation(format!(
                    "input_ids contains out-of-vocabulary token id {id}; \
                     valid range is [0, {vocab_size})"
                )));
            }
        }
    }
    Ok(())
}

/// OpenAI's context-overflow sentence; clients match on it.
fn context_length_sentence(max_tokens: u64) -> String {
    format!("This model's maximum context length is {max_tokens} tokens on this server.")
}

/// The rejection of an input over `max_req_input_len`, or `None`. A multimodal
/// prompt is never truncated (a cut can split placeholder tokens).
pub(crate) fn input_length_error(
    input_len: usize,
    limits: &Limits,
    is_multimodal: bool,
) -> Option<String> {
    let max_req_input_len = limits.max_req_input_len;
    if input_len as u64 <= max_req_input_len {
        return None;
    }
    let error = format!(
        "Input length ({input_len} tokens) exceeds the maximum allowed length \
         ({max_req_input_len} tokens). {} ",
        context_length_sentence(max_req_input_len)
    );
    Some(if is_multimodal {
        format!(
            "Multimodal prompt is too long after expanding multimodal tokens. \
             {error}Use a shorter input."
        )
    } else {
        format!("{error}Use a shorter input or enable --allow-auto-truncate.")
    })
}

/// The context-window checks that need the tokenized length, mirroring Python
/// `TokenizerManager._validate_one_request`: the input must fit
/// `max_req_input_len`, and then input + reserved slots + the client's own
/// `max_new_tokens` must fit `max_req_len`. Without them the scheduler
/// silently clamps and the client gets a 200 with a truncated completion
/// instead of an actionable 400. A defaulted `max_new_tokens` (see
/// [`GenerateRequest::max_new_tokens_defaulted`]) is left to that clamp.
///
/// Under `allow_auto_truncate` both clamp instead of rejecting.
pub(super) fn check_total_tokens(g: &mut GenerateRequest, limits: &Limits) -> Result<(), Error> {
    // Input length first, and unconditionally: `max_new_tokens: null` means "no
    // cap", which must not disable this. The MM worker already checked a
    // multimodal prompt.
    if let Some(input_ids) = g.input_ids.as_mut()
        && let Some(error) = input_length_error(input_ids.len(), limits, false)
    {
        if !limits.allow_auto_truncate {
            return Err(Error::Validation(error));
        }
        input_ids.truncate(limits.max_req_input_len as usize);
    }
    // Python counts the reserved slots as part of the input in the total, so a
    // request can be rejected for them even when the prompt alone fits.
    let input_token_num =
        g.input_ids.as_ref().map_or(0, |ids| ids.len()) as u64 + limits.num_reserved_tokens;

    if g.max_new_tokens_defaulted {
        return Ok(()); // not the client's value → the scheduler clamps it
    }
    let Some(max_new_tokens) = g.sampling_params.max_new_tokens else {
        return Ok(()); // no cap requested → nothing to add to the input length
    };
    let total = input_token_num.saturating_add(max_new_tokens.max(0) as u64);
    if total <= limits.max_req_len {
        return Ok(());
    }
    if !limits.allow_auto_truncate {
        return Err(Error::Validation(format!(
            "Requested token count exceeds the model's maximum context length. {} You \
             requested a total of {total} tokens: {input_token_num} \
             tokens from the input messages and {max_new_tokens} tokens for the \
             completion. Please reduce the number of tokens in the input messages or \
             the completion to fit within the limit.",
            context_length_sentence(limits.max_req_len)
        )));
    }
    let context_max_new_tokens = limits.max_req_len.saturating_sub(input_token_num) as i64;
    // Re-check what the clamp can break. `verify` already ran (in Normalizing), so
    // lowering `max_new_tokens` here can leave `min_new_tokens > max_new_tokens` —
    // and `is_normalized: true` stops the scheduler from re-verifying, so nothing
    // downstream would catch it. Python validates before it verifies; we can't
    // reorder the FSM, so we re-assert the one invariant the clamp can violate.
    if g.sampling_params.min_new_tokens > context_max_new_tokens {
        return Err(Error::Validation(format!(
            "min_new_tokens must be in [0, max_new_tokens({context_max_new_tokens})], got {}",
            g.sampling_params.min_new_tokens
        )));
    }
    g.sampling_params.max_new_tokens = Some(context_max_new_tokens);
    Ok(())
}
