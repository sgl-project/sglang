//! Request validation for scheduler intake.

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
    // client-supplied rid is therefore a per-step cost, not a one-off. Python's is
    // a 32-byte uuid hex, so this is generous.
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

    // Client-supplied token ids must be in-vocabulary: an out-of-range id
    // reaches the embedding lookup and kills the scheduler process, so 400
    // here instead — mirroring the Python `TokenizerManager` validation.
    if let RequestKind::Generate(g) = &req.kind {
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
    if let RequestKind::Detokenize { token_ids } = &req.kind {
        for &id in token_ids {
            if u32::try_from(id).is_err() {
                return Err(Error::Validation(format!("Token ID {id} is out of range")));
            }
        }
    }

    // The scheduler only computes hidden states when launched for it, so without
    // this the request would 200 with `meta_info.hidden_states` silently absent
    // (Python `TokenizerManager._validate_one_request`).
    if !limits.enable_return_hidden_states
        && matches!(&req.kind, RequestKind::Generate(g) if g.return_hidden_states)
    {
        return Err(Error::Validation(
            "The server is not configured to return the hidden states. \
             Please set `--enable-return-hidden-states` to enable this feature."
                .into(),
        ));
    }

    Ok(())
}

/// The context-window checks that need the tokenized length, mirroring Python
/// `TokenizerManager._validate_one_request`: the input alone must fit, and then
/// input + `max_new_tokens` must fit. Without them the scheduler silently clamps
/// and the client gets a 200 with a truncated completion instead of an actionable
/// 400.
///
/// Under `allow_auto_truncate` both clamp instead of rejecting — the launch flag
/// opted into that.
pub(super) fn check_total_tokens(g: &mut GenerateRequest, limits: &Limits) -> Result<(), Error> {
    let max_req_len = limits.context_len;
    // Python counts the reserved slots as part of the input, so a request can be
    // rejected for them even when the prompt alone fits.
    let input_len =
        g.input_ids.as_ref().map_or(0, |ids| ids.len()) as u64 + limits.num_reserved_tokens;

    // Input length first, and unconditionally: `max_new_tokens: null` means "no
    // cap", which must not disable this. Python's comparison is `>=` — a prompt
    // that exactly fills the window leaves no room to generate.
    if input_len >= max_req_len {
        if !limits.allow_auto_truncate {
            return Err(Error::Validation(format!(
                "The input ({input_len} tokens) is longer than the model's context \
                 length ({max_req_len} tokens)."
            )));
        }
        if let Some(ids) = &mut g.input_ids {
            ids.truncate(max_req_len as usize);
        }
    }
    let input_len =
        g.input_ids.as_ref().map_or(0, |ids| ids.len()) as u64 + limits.num_reserved_tokens;

    let Some(max_new_tokens) = g.sampling_params.max_new_tokens else {
        return Ok(()); // no cap requested → nothing to add to the input length
    };
    let total = input_len.saturating_add(max_new_tokens.max(0) as u64);
    if total <= max_req_len {
        return Ok(());
    }
    if !limits.allow_auto_truncate {
        return Err(Error::Validation(format!(
            "Requested token count exceeds the model's maximum context length of \
             {max_req_len} tokens. You requested a total of {total} tokens: {input_len} \
             tokens from the input messages and {max_new_tokens} tokens for the \
             completion. Please reduce the number of tokens in the input messages or \
             the completion to fit within the limit."
        )));
    }
    let clamped = max_req_len.saturating_sub(input_len) as i64;
    // Re-check what the clamp can break. `verify` already ran (in Normalizing), so
    // lowering `max_new_tokens` here can leave `min_new_tokens > max_new_tokens` —
    // and `is_normalized: true` stops the scheduler from re-verifying, so nothing
    // downstream would catch it. Python validates before it verifies; we can't
    // reorder the FSM, so we re-assert the one invariant the clamp can violate.
    if g.sampling_params.min_new_tokens > clamped {
        return Err(Error::Validation(format!(
            "min_new_tokens must be in [0, max_new_tokens({clamped})], got {}",
            g.sampling_params.min_new_tokens
        )));
    }
    g.sampling_params.max_new_tokens = Some(clamped);
    Ok(())
}
