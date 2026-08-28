//! Transport-neutral OpenAI model-card shaping.

use crate::api_server::core::openai::unix_seconds_u32;
use crate::message::config::ServerArgs;

pub(crate) fn model_card(server_args: &ServerArgs) -> serde_json::Value {
    let name = &server_args.served_model_name;
    serde_json::json!({
        "id": name,
        "object": "model",
        "created": unix_seconds_u32(),
        "owned_by": "sglang",
        "root": name,
        "parent": serde_json::Value::Null,
        "max_model_len": server_args.model_config.context_len,
    })
}
