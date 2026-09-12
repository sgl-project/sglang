// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Which HTTP request surface a proxied generation request arrived on, and
//! how the router's per-model controls (`--max-output-tokens`,
//! `--override-sampling-params`) read and write it.
//!
//! The two surfaces place the SAME controls in different spots: the OpenAI
//! chat surface carries them as top-level keys, while sglang's native
//! `/generate` nests them under `sampling_params.*` (and names the output
//! budget `max_new_tokens`). Every `SamplingField::wire_name()` is
//! byte-identical to the corresponding `SamplingParams` field, so the
//! translation is exact; only the output budget renames, and it is
//! READ-only on `Generate` (enforced, never injected — see
//! [`Surface::write_injections`]).

use super::chat::{lax_number, RequestProbe};
use crate::config::SamplingField;
use serde_json::{Map, Number, Value};

/// The request surface a generation request arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Surface {
    /// `POST /v1/chat/completions` — OpenAI chat shape.
    Chat,
    /// `POST /generate` — sglang-native shape (`text` / `input_ids` +
    /// `sampling_params`).
    Generate,
}

impl Surface {
    /// The path the request is proxied to on the worker. Today each surface
    /// forwards to its own spelling — the engine serves both.
    pub(crate) fn upstream_path(self) -> &'static str {
        match self {
            Surface::Chat => "/v1/chat/completions",
            Surface::Generate => "/generate",
        }
    }

    /// The request's explicit output budget, if it set one. `Chat` mirrors
    /// the engine's `max_completion_tokens or max_tokens` resolution
    /// (Python `or`: an explicit numeric `0` is falsy); `Generate` reads
    /// `sampling_params.max_new_tokens`. Note the asymmetric null handling:
    /// on `Chat` an explicit `null` deserializes to absent (which the cap
    /// then defaults), but on `Generate` the value is returned AS `null` —
    /// the engine reads an explicit null `max_new_tokens` as UNBOUNDED
    /// (`init_req_max_new_tokens` maps it to `1 << 30`), not as the 128
    /// dataclass default an absent key gets, so the cap check must see it.
    pub(crate) fn read_output_budget<'a>(&self, probe: &'a RequestProbe) -> Option<&'a Value> {
        match self {
            Surface::Chat => probe
                .max_completion_tokens
                .as_ref()
                .filter(|v| lax_number(v) != Some(0.0))
                .or(probe.max_tokens.as_ref()),
            Surface::Generate => probe
                .sampling_params_obj()
                .and_then(|o| o.get("max_new_tokens")),
        }
    }

    /// The request's value for one sampling parameter, if it set one:
    /// top-level on `Chat`, nested under `sampling_params` on `Generate`.
    /// Explicit `null` is treated as absent on both.
    pub(crate) fn read_sampling_field<'a>(
        &self,
        probe: &'a RequestProbe,
        field: SamplingField,
    ) -> Option<&'a Value> {
        match self {
            Surface::Chat => probe.sampling_field(field),
            Surface::Generate => probe
                .sampling_params_obj()
                .and_then(|o| o.get(field.wire_name()))
                .filter(|v| !v.is_null()),
        }
    }

    /// The key [`super::chat::output_budget_action`] names in its over-cap
    /// 400, so the message points at a field the client actually sent.
    pub(crate) fn budget_error_key(self) -> &'static str {
        match self {
            Surface::Chat => "max_tokens",
            Surface::Generate => "sampling_params.max_new_tokens",
        }
    }

    /// Write the router's injections into the outgoing body: the output
    /// budget default and the pinned sampling params. Top-level on `Chat`;
    /// nested under a created-if-absent `sampling_params` object on
    /// `Generate`.
    ///
    /// The budget is IGNORED on `Generate`: an absent
    /// `sampling_params.max_new_tokens` there takes the engine's
    /// `SamplingParams` dataclass default (128), already far under any sane
    /// cap, so injecting the cap would turn a ceiling into a floor and RAISE
    /// per-request output. `Chat` genuinely needs the injection — an absent
    /// budget maps to an explicit unbounded `max_new_tokens=None`.
    pub(crate) fn write_injections(
        self,
        obj: &mut Map<String, Value>,
        budget: Option<u64>,
        sampling: &[(SamplingField, Number)],
    ) {
        match self {
            Surface::Chat => {
                if let Some(cap) = budget {
                    // Caller passes `Some` only when the request set no output
                    // budget at all, so this never overrides a client value.
                    obj.insert("max_tokens".to_string(), Value::Number(cap.into()));
                }
                for (field, value) in sampling {
                    obj.insert(field.wire_name().to_string(), Value::Number(value.clone()));
                }
            }
            Surface::Generate => {
                if sampling.is_empty() {
                    return;
                }
                let sp = obj
                    .entry("sampling_params".to_string())
                    .or_insert_with(|| Value::Object(Map::new()));
                if !sp.is_object() {
                    // Explicit `null` is treated as absent; a present
                    // non-object is 400'd before dispatch, so nothing else
                    // can reach here.
                    *sp = Value::Object(Map::new());
                }
                let sp = sp
                    .as_object_mut()
                    .expect("sampling_params object just ensured");
                for (field, value) in sampling {
                    sp.insert(field.wire_name().to_string(), Value::Number(value.clone()));
                }
            }
        }
    }
}
