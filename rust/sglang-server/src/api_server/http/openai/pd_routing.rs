//! SGLang P/D extensions to Dynamo's standard OpenAI request types.

use serde::Deserialize;
use sglang_api_types::api::v1::{Int64OrList, OptionalInt64OrList, StringOrList};

use crate::message::request::{BootstrapColumns, normalize_bootstrap_columns};
use crate::message::types::OneOrMany;
use crate::message::wire;
use crate::utils::error::Error;

/// Python's OpenAI PDRoutingFields, using generated API carriers. Unlike the
/// native API, OpenAI allows null list elements only for bootstrap_port.
#[derive(Deserialize)]
pub(super) struct PDRoutingFields {
    bootstrap_host: Option<StringOrList>,
    bootstrap_port: Option<OptionalInt64OrList>,
    bootstrap_room: Option<Int64OrList>,
    routed_dp_rank: Option<i64>,
    disagg_prefill_dp_rank: Option<i64>,
}

/// Validated per-prompt columns, ready for admission.
pub(super) struct NormalizedPDRouting {
    pub(super) bootstrap: BootstrapColumns,
    pub(super) routed_dp_rank: Option<i64>,
    pub(super) disagg_prefill_dp_rank: Option<i64>,
}

impl PDRoutingFields {
    /// Normalize bootstrap fields per engine request (prompts × choices, in
    /// Python's sample-major flat order) and bound the host-string clones.
    /// DP hints remain scalar.
    pub(super) fn normalize(
        self,
        prompt_count: usize,
        choices_per_prompt: usize,
    ) -> Result<NormalizedPDRouting, Error> {
        let hosts = self
            .bootstrap_host
            .and_then(wire::string_or_list)
            .map(|hosts| match hosts {
                OneOrMany::One(host) => OneOrMany::One(Some(host)),
                OneOrMany::Many(hosts) => OneOrMany::Many(hosts.into_iter().map(Some).collect()),
            });
        let rooms = self
            .bootstrap_room
            .and_then(wire::int64_or_list)
            .map(|rooms| match rooms {
                OneOrMany::One(room) => OneOrMany::One(Some(room)),
                OneOrMany::Many(rooms) => OneOrMany::Many(rooms.into_iter().map(Some).collect()),
            });
        let bootstrap = normalize_bootstrap_columns(
            hosts,
            self.bootstrap_port.and_then(wire::optional_int64_or_list),
            rooms,
            prompt_count,
            choices_per_prompt,
        )?;
        Ok(NormalizedPDRouting {
            bootstrap,
            routed_dp_rank: self.routed_dp_rank,
            disagg_prefill_dp_rank: self.disagg_prefill_dp_rank,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::PDRoutingFields;
    use serde_json::json;

    #[test]
    fn list_routing_preserves_per_prompt_metadata() {
        // mini_lb injects per-prompt scalars for text batches; the GPU fixture
        // does not cover lists or distinguish nonzero DP hints in its
        // single-rank topology. A per-prompt list is shared by that prompt's
        // choices (Python replicates the list × n over its sample-major flat
        // order, so flat % batch_size is the prompt index).
        let fields: PDRoutingFields = serde_json::from_value(json!({
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [9007199254740993_i64, 9007199254740995_i64],
            "routed_dp_rank": 1,
            "disagg_prefill_dp_rank": 3,
        }))
        .unwrap();
        let routing = fields.normalize(2, 2).unwrap();
        assert_eq!(
            routing.bootstrap.bootstrap_hosts,
            vec![
                Some("prefill-a".to_owned()),
                Some("prefill-b".to_owned()),
                Some("prefill-a".to_owned()),
                Some("prefill-b".to_owned())
            ]
        );
        assert_eq!(
            routing.bootstrap.bootstrap_ports,
            vec![Some(8998), None, Some(8998), None]
        );
        assert_eq!(
            routing.bootstrap.bootstrap_rooms,
            vec![
                Some(9007199254740993),
                Some(9007199254740995),
                Some(9007199254740993),
                Some(9007199254740995)
            ]
        );
        assert_eq!(routing.routed_dp_rank, Some(1));
        assert_eq!(routing.disagg_prefill_dp_rank, Some(3));
    }

    /// The PD router sizes chat room lists by `n` (`get_chat_batch_size`
    /// returns `n`), so chat `n > 1` arrives with an n-element
    /// `bootstrap_room` list. Python accepts it (one room per choice); the
    /// Rust worker must too, or every routed `n > 1` chat request 400s.
    #[test]
    fn per_choice_room_list_is_accepted_for_chat_n() {
        let fields: PDRoutingFields = serde_json::from_value(json!({
            "bootstrap_host": ["h0", "h1"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [100, 101],
        }))
        .unwrap();
        let routing = fields.normalize(1, 2).unwrap();
        assert_eq!(
            routing.bootstrap.bootstrap_rooms,
            vec![Some(100), Some(101)]
        );
        assert_eq!(
            routing.bootstrap.bootstrap_hosts,
            vec![Some("h0".to_owned()), Some("h1".to_owned())]
        );
        assert_eq!(routing.bootstrap.bootstrap_ports, vec![Some(8998), None]);
    }

    /// A scalar room fans out over prompts × choices in Python's sample-major
    /// flat order (`room + flat`, `flat = choice * prompt_count + prompt`), so
    /// a mixed Rust/Python prefill-decode pair rendezvous every choice.
    #[test]
    fn scalar_room_fans_out_across_prompts_and_choices() {
        let fields: PDRoutingFields =
            serde_json::from_value(json!({"bootstrap_room": 100})).unwrap();
        let routing = fields.normalize(2, 3).unwrap();
        assert_eq!(
            routing.bootstrap.bootstrap_rooms,
            vec![
                Some(100), // choice 0: prompts 0, 1
                Some(101),
                Some(102), // choice 1: prompts 0, 1
                Some(103),
                Some(104), // choice 2: prompts 0, 1
                Some(105)
            ]
        );

        let fields: PDRoutingFields =
            serde_json::from_value(json!({"bootstrap_host": "h", "bootstrap_room": 7})).unwrap();
        let routing = fields.normalize(1, 1).unwrap();
        assert_eq!(routing.bootstrap.bootstrap_rooms, vec![Some(7)]);
        assert_eq!(
            routing.bootstrap.bootstrap_hosts,
            vec![Some("h".to_owned())]
        );
    }

    /// A list matching neither the prompt count nor the per-choice count is
    /// still a 400 — the two router conventions are the only accepted shapes.
    #[test]
    fn room_list_of_unrecognized_length_is_rejected() {
        let fields: PDRoutingFields =
            serde_json::from_value(json!({"bootstrap_room": [1, 2, 3]})).unwrap();
        let error = fields.normalize(2, 2).err().unwrap().to_string();
        assert!(error.contains("bootstrap_room"), "{error}");
        assert!(error.contains("neither the prompt count"), "{error}");

        // With one choice per prompt, only the per-prompt length is accepted.
        let fields: PDRoutingFields =
            serde_json::from_value(json!({"bootstrap_room": [1, 2, 3]})).unwrap();
        let error = fields.normalize(2, 1).err().unwrap().to_string();
        assert!(error.contains("bootstrap_room"), "{error}");
    }

    #[test]
    fn null_list_elements_are_rejected_except_for_ports() {
        for body in [
            r#"{"bootstrap_host":[null]}"#,
            r#"{"bootstrap_room":[null]}"#,
        ] {
            assert!(serde_json::from_str::<PDRoutingFields>(body).is_err());
        }
    }

    #[test]
    fn list_hosts_cannot_bypass_choice_clone_budget() {
        // 263173 bytes fit for one choice, but 255 choices exceed 64 MiB.
        let fields: PDRoutingFields =
            serde_json::from_value(json!({"bootstrap_host": ["x".repeat(263173)]})).unwrap();
        let error = fields.normalize(1, 255).err().unwrap().to_string();
        assert!(error.contains("bootstrap_host"), "{error}");
        assert!(error.contains("would allocate more than"), "{error}");
    }
}
