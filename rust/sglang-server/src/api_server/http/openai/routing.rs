//! SGLang P/D extensions to Dynamo's standard OpenAI request types.

use serde::Deserialize;
use sglang_api_types::api::v1::{Int64OrList, OptionalInt64OrList, StringOrList};

use crate::message::request::{BootstrapColumns, GenerateRequest, normalize_bootstrap_columns};
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
    bootstrap: BootstrapColumns,
    routed_dp_rank: Option<i64>,
    disagg_prefill_dp_rank: Option<i64>,
}

impl PDRoutingFields {
    /// Convert the wire carriers and normalize their bootstrap columns once,
    /// retaining the scalar DP hints for each submitted choice.
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
        Ok(PDRouting {
            bootstrap: normalize_bootstrap_columns(
                hosts,
                self.bootstrap_port.and_then(wire::optional_int64_or_list),
                rooms,
                prompt_count,
                n,
            )?,
            routed_dp_rank: self.routed_dp_rank,
            disagg_prefill_dp_rank: self.disagg_prefill_dp_rank,
        })
    }
}

impl PDRouting {
    /// Like Python's parallel-sampling dispatch, each prompt's choices share
    /// its routing metadata; only request IDs and response indices fan out.
    pub(super) fn apply(&self, request: &mut GenerateRequest, prompt_index: usize) {
        request.bootstrap_host = self.bootstrap.bootstrap_hosts[prompt_index].clone();
        request.bootstrap_port = self.bootstrap.bootstrap_ports[prompt_index];
        request.bootstrap_room = self.bootstrap.bootstrap_rooms[prompt_index];
        request.routed_dp_rank = self.routed_dp_rank;
        request.disagg_prefill_dp_rank = self.disagg_prefill_dp_rank;
    }
}
