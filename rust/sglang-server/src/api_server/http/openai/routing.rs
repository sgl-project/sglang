//! SGLang P/D extensions to Dynamo's standard OpenAI request types.

use serde::Deserialize;
use sglang_api_types::api::v1::{Int64OrList, OptionalInt64OrList, StringOrList};

use crate::message::request::{
    BootstrapColumns, check_broadcast_budget, normalize_bootstrap_columns,
};
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
pub(super) struct PDRouting {
    pub(super) bootstrap: BootstrapColumns,
    pub(super) routed_dp_rank: Option<i64>,
    pub(super) disagg_prefill_dp_rank: Option<i64>,
}

impl PDRoutingFields {
    /// Normalize bootstrap fields per prompt and bound the handlers' host clones
    /// across choices. DP hints remain scalar.
    pub(super) fn into_routing(self, prompt_count: usize, n: usize) -> Result<PDRouting, Error> {
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
        )?;
        if n > 1 {
            check_broadcast_budget(
                bootstrap
                    .bootstrap_hosts
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(String::len)
                    .sum(),
                n,
                "bootstrap_host",
            )?;
        }
        Ok(PDRouting {
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
        // mini_lb injects scalars; the GPU fixture does not cover lists or
        // distinguish nonzero DP hints in its single-rank topology.
        let fields: PDRoutingFields = serde_json::from_value(json!({
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [9007199254740993_i64, 9007199254740995_i64],
            "routed_dp_rank": 1,
            "disagg_prefill_dp_rank": 3,
        }))
        .unwrap();
        let routing = fields.into_routing(2, 2).unwrap();
        assert_eq!(
            routing.bootstrap.bootstrap_hosts,
            vec![Some("prefill-a".to_owned()), Some("prefill-b".to_owned())]
        );
        assert_eq!(routing.bootstrap.bootstrap_ports, vec![Some(8998), None]);
        assert_eq!(
            routing.bootstrap.bootstrap_rooms,
            vec![Some(9007199254740993), Some(9007199254740995)]
        );
        assert_eq!(routing.routed_dp_rank, Some(1));
        assert_eq!(routing.disagg_prefill_dp_rank, Some(3));
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
        let error = fields.into_routing(1, 255).err().unwrap().to_string();
        assert!(error.contains("bootstrap_host"), "{error}");
        assert!(error.contains("would allocate more than"), "{error}");
    }
}
