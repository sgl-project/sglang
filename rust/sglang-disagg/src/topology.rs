//! Bootstrap `/route` HTTP bodies (`common/conn.py` `register_to_bootstrap`,
//! `_handle_route_put`, `_handle_route_get`). All bodies are
//! `json.dumps` with DEFAULT separators — use [`dumps_default`].
//!
//! Three shapes:
//! - PUT body: the 21-key register payload (decode_allocation_policy is
//!   appended after the literal, so it serializes LAST).
//! - GET rank-query response: `PrefillRankInfo` asdict — exactly
//!   `{"rank_ip", "rank_port"}`.
//! - GET all-(-1) response: `PrefillServerInfo` asdict in dataclass order,
//!   with `dsv41_spec_layout`/`kv_layout` popped when None. The Rust
//!   registry (`rust/sglang-server` `bootstrap.rs`) serves a 13-key subset;
//!   the parser tolerates both.

use crate::Result;
use crate::error::Error;
use crate::json::JVal;

/// PUT `/route` body fields (`register_to_bootstrap`).
#[derive(Clone, Debug, PartialEq)]
pub struct RegisterParams {
    pub attn_tp_size: i64,
    pub attn_tp_rank: i64,
    pub attn_cp_size: i64,
    pub attn_cp_rank: i64,
    pub attn_dp_size: i64,
    pub attn_dp_rank: i64,
    pub pp_size: i64,
    pub pp_rank: i64,
    pub system_dp_size: i64,
    pub system_dp_rank: i64,
    pub rank_ip: String,
    pub rank_port: i64,
    pub page_size: Option<i64>,
    pub kv_cache_dtype: Option<String>,
    pub kv_layout: Option<String>,
    /// Arbitrary JSON object (DeepSeek-V4.1 DSpark PD layout) or null.
    pub dsv41_spec_layout: Option<JVal>,
    pub load_balance_method: Option<String>,
    pub enable_dsa_cache_layer_split: bool,
    pub enable_kv_cache_sharding: bool,
    pub prefill_http_port: Option<i64>,
    /// Appended after the payload literal — serializes last.
    pub decode_allocation_policy: String,
}

fn opt_str(v: &Option<String>) -> JVal {
    match v {
        Some(s) => JVal::Str(s.clone()),
        None => JVal::Null,
    }
}

fn opt_int(v: &Option<i64>) -> JVal {
    match v {
        Some(i) => JVal::Int(*i),
        None => JVal::Null,
    }
}

/// The PUT body as an order-preserving JSON object (Python dict insertion
/// order, `decode_allocation_policy` assigned last).
pub fn register_body(p: &RegisterParams) -> JVal {
    JVal::Obj(vec![
        ("attn_tp_size".to_string(), JVal::Int(p.attn_tp_size)),
        ("attn_tp_rank".to_string(), JVal::Int(p.attn_tp_rank)),
        ("attn_cp_size".to_string(), JVal::Int(p.attn_cp_size)),
        ("attn_cp_rank".to_string(), JVal::Int(p.attn_cp_rank)),
        ("attn_dp_size".to_string(), JVal::Int(p.attn_dp_size)),
        ("attn_dp_rank".to_string(), JVal::Int(p.attn_dp_rank)),
        ("pp_size".to_string(), JVal::Int(p.pp_size)),
        ("pp_rank".to_string(), JVal::Int(p.pp_rank)),
        ("system_dp_size".to_string(), JVal::Int(p.system_dp_size)),
        ("system_dp_rank".to_string(), JVal::Int(p.system_dp_rank)),
        ("rank_ip".to_string(), JVal::Str(p.rank_ip.clone())),
        ("rank_port".to_string(), JVal::Int(p.rank_port)),
        ("page_size".to_string(), opt_int(&p.page_size)),
        ("kv_cache_dtype".to_string(), opt_str(&p.kv_cache_dtype)),
        ("kv_layout".to_string(), opt_str(&p.kv_layout)),
        (
            "dsv41_spec_layout".to_string(),
            p.dsv41_spec_layout.clone().unwrap_or(JVal::Null),
        ),
        (
            "load_balance_method".to_string(),
            opt_str(&p.load_balance_method),
        ),
        (
            "enable_dsa_cache_layer_split".to_string(),
            JVal::Bool(p.enable_dsa_cache_layer_split),
        ),
        (
            "enable_kv_cache_sharding".to_string(),
            JVal::Bool(p.enable_kv_cache_sharding),
        ),
        (
            "prefill_http_port".to_string(),
            opt_int(&p.prefill_http_port),
        ),
        (
            "decode_allocation_policy".to_string(),
            JVal::Str(p.decode_allocation_policy.clone()),
        ),
    ])
}

fn need<'a>(v: &'a JVal, key: &str) -> Result<&'a JVal> {
    v.get(key)
        .ok_or_else(|| Error::Codec(format!("register body: missing key {:?}", key)))
}

fn get_int(v: &JVal, key: &str) -> Result<i64> {
    need(v, key)?
        .as_int()
        .ok_or_else(|| Error::Codec(format!("register body: {:?} is not an int", key)))
}

fn get_opt_int(v: &JVal, key: &str) -> Result<Option<i64>> {
    match v.get(key) {
        None | Some(JVal::Null) => Ok(None),
        Some(j) => j
            .as_int()
            .map(Some)
            .ok_or_else(|| Error::Codec(format!("register body: {:?} is not an int", key))),
    }
}

fn get_str(v: &JVal, key: &str) -> Result<String> {
    need(v, key)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::Codec(format!("register body: {:?} is not a string", key)))
}

fn get_opt_str(v: &JVal, key: &str) -> Result<Option<String>> {
    match v.get(key) {
        None | Some(JVal::Null) => Ok(None),
        Some(JVal::Str(s)) => Ok(Some(s.clone())),
        Some(_) => Err(Error::Codec(format!(
            "register body: {:?} is not a string",
            key
        ))),
    }
}

fn get_bool_default(v: &JVal, key: &str, default: bool) -> Result<bool> {
    match v.get(key) {
        None | Some(JVal::Null) => Ok(default),
        Some(JVal::Bool(b)) => Ok(*b),
        Some(_) => Err(Error::Codec(format!(
            "register body: {:?} is not a bool",
            key
        ))),
    }
}

/// `_handle_route_put` body decode: `data["k"]` for the required keys
/// (KeyError -> error here), `data.get` with Python's defaults for the rest.
/// `dsv41_spec_layout` stays a raw value (None when JSON null).
pub fn parse_register_body(v: &JVal) -> Result<RegisterParams> {
    Ok(RegisterParams {
        attn_tp_size: get_int(v, "attn_tp_size")?,
        attn_tp_rank: get_int(v, "attn_tp_rank")?,
        attn_cp_size: get_int(v, "attn_cp_size")?,
        attn_cp_rank: get_int(v, "attn_cp_rank")?,
        attn_dp_size: get_int(v, "attn_dp_size")?,
        attn_dp_rank: get_int(v, "attn_dp_rank")?,
        pp_size: get_int(v, "pp_size")?,
        pp_rank: get_int(v, "pp_rank")?,
        system_dp_size: get_int(v, "system_dp_size")?,
        system_dp_rank: get_int(v, "system_dp_rank")?,
        rank_ip: get_str(v, "rank_ip")?,
        rank_port: get_int(v, "rank_port")?,
        page_size: get_opt_int(v, "page_size")?,
        kv_cache_dtype: get_opt_str(v, "kv_cache_dtype")?,
        kv_layout: get_opt_str(v, "kv_layout")?,
        dsv41_spec_layout: match v.get("dsv41_spec_layout") {
            None | Some(JVal::Null) => None,
            Some(j) => Some(j.clone()),
        },
        load_balance_method: get_opt_str(v, "load_balance_method")?,
        enable_dsa_cache_layer_split: get_bool_default(v, "enable_dsa_cache_layer_split", false)?,
        enable_kv_cache_sharding: get_bool_default(v, "enable_kv_cache_sharding", false)?,
        prefill_http_port: get_opt_int(v, "prefill_http_port")?,
        decode_allocation_policy: match v.get("decode_allocation_policy") {
            None | Some(JVal::Null) => "early".to_string(),
            Some(JVal::Str(s)) => s.clone(),
            Some(_) => {
                return Err(Error::Codec(
                    "register body: \"decode_allocation_policy\" is not a string".into(),
                ));
            }
        },
    })
}

/// GET `/route` rank-query response (`PrefillRankInfo` asdict).
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillRankInfo {
    pub rank_ip: String,
    pub rank_port: i64,
}

/// `dataclasses.asdict(PrefillRankInfo)` — exactly two keys.
pub fn rank_info_body(info: &PrefillRankInfo) -> JVal {
    JVal::Obj(vec![
        ("rank_ip".to_string(), JVal::Str(info.rank_ip.clone())),
        ("rank_port".to_string(), JVal::Int(info.rank_port)),
    ])
}

/// Rank-response decode: `data["rank_ip"]`, `data["rank_port"]` (KeyError ->
/// error).
pub fn parse_rank_info(v: &JVal) -> Result<PrefillRankInfo> {
    Ok(PrefillRankInfo {
        rank_ip: get_str(v, "rank_ip")?,
        rank_port: get_int(v, "rank_port")?,
    })
}

/// GET `/route` all-(-1) response (`PrefillServerInfo`). Wire-relevant
/// fields only; the decode-side `target_*`/`required_*` fields serialize as
/// nulls from the aiohttp server and are absent from the Rust registry — the
/// parser ignores them.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillServerInfo {
    pub attn_tp_size: i64,
    pub attn_cp_size: i64,
    pub dp_size: i64,
    pub pp_size: i64,
    pub page_size: Option<i64>,
    pub kv_cache_dtype: Option<String>,
    pub follow_bootstrap_room: bool,
    pub enable_dsa_cache_layer_split: bool,
    pub dsv41_spec_layout: Option<JVal>,
    pub kv_layout: Option<String>,
    pub decode_allocation_policy: String,
    pub enable_kv_cache_sharding: bool,
    pub prefill_http_port: Option<i64>,
}

/// `dataclasses.asdict(PrefillServerInfo)` in dataclass declaration order,
/// with `dsv41_spec_layout`/`kv_layout` popped when None
/// (`_handle_route_get`). The six decode-side fields
/// (`target_tp_rank`, `target_tp_ranks`, `target_cp_ranks`,
/// `target_pp_ranks`, `required_dst_info_num`,
/// `required_prefill_response_num`) are always null on the wire.
pub fn server_info_body(info: &PrefillServerInfo) -> JVal {
    let mut kvs = vec![
        ("attn_tp_size".to_string(), JVal::Int(info.attn_tp_size)),
        ("attn_cp_size".to_string(), JVal::Int(info.attn_cp_size)),
        ("dp_size".to_string(), JVal::Int(info.dp_size)),
        ("pp_size".to_string(), JVal::Int(info.pp_size)),
        ("page_size".to_string(), opt_int(&info.page_size)),
        ("kv_cache_dtype".to_string(), opt_str(&info.kv_cache_dtype)),
        (
            "follow_bootstrap_room".to_string(),
            JVal::Bool(info.follow_bootstrap_room),
        ),
        (
            "enable_dsa_cache_layer_split".to_string(),
            JVal::Bool(info.enable_dsa_cache_layer_split),
        ),
        (
            "dsv41_spec_layout".to_string(),
            info.dsv41_spec_layout.clone().unwrap_or(JVal::Null),
        ),
        ("kv_layout".to_string(), opt_str(&info.kv_layout)),
        (
            "decode_allocation_policy".to_string(),
            JVal::Str(info.decode_allocation_policy.clone()),
        ),
        (
            "enable_kv_cache_sharding".to_string(),
            JVal::Bool(info.enable_kv_cache_sharding),
        ),
        (
            "prefill_http_port".to_string(),
            opt_int(&info.prefill_http_port),
        ),
        ("target_tp_rank".to_string(), JVal::Null),
        ("target_tp_ranks".to_string(), JVal::Null),
        ("target_cp_ranks".to_string(), JVal::Null),
        ("target_pp_ranks".to_string(), JVal::Null),
        ("required_dst_info_num".to_string(), JVal::Null),
        ("required_prefill_response_num".to_string(), JVal::Null),
    ];
    // `_handle_route_get` pops these when None — remove from the END first so
    // indices stay valid (they are not adjacent; find by key).
    if info.dsv41_spec_layout.is_none() {
        kvs.retain(|(k, _)| k != "dsv41_spec_layout");
    }
    if info.kv_layout.is_none() {
        kvs.retain(|(k, _)| k != "kv_layout");
    }
    JVal::Obj(kvs)
}

/// Parse a GET all-(-1) response. Tolerates the 13-key Rust-registry
/// subset; Python dataclass defaults fill the optional fields. Unknown keys
/// (a newer SGLang's extra fields) are ignored.
pub fn parse_server_info(v: &JVal) -> Result<PrefillServerInfo> {
    v.as_obj()
        .ok_or_else(|| Error::Codec("prefill row: not an object".into()))?;
    let follow = match v.get("follow_bootstrap_room") {
        None => {
            return Err(Error::Codec(
                "prefill row: missing \"follow_bootstrap_room\"".into(),
            ));
        }
        Some(JVal::Bool(b)) => *b,
        Some(_) => {
            return Err(Error::Codec(
                "prefill row: \"follow_bootstrap_room\" is not a bool".into(),
            ));
        }
    };
    Ok(PrefillServerInfo {
        attn_tp_size: get_int(v, "attn_tp_size")?,
        attn_cp_size: get_int(v, "attn_cp_size")?,
        dp_size: get_int(v, "dp_size")?,
        pp_size: get_int(v, "pp_size")?,
        page_size: get_opt_int(v, "page_size")?,
        kv_cache_dtype: get_opt_str(v, "kv_cache_dtype")?,
        follow_bootstrap_room: follow,
        enable_dsa_cache_layer_split: get_bool_default(v, "enable_dsa_cache_layer_split", false)?,
        dsv41_spec_layout: match v.get("dsv41_spec_layout") {
            None | Some(JVal::Null) => None,
            Some(j) => Some(j.clone()),
        },
        kv_layout: get_opt_str(v, "kv_layout")?,
        decode_allocation_policy: match v.get("decode_allocation_policy") {
            None | Some(JVal::Null) => "early".to_string(),
            Some(JVal::Str(s)) => s.clone(),
            Some(_) => {
                return Err(Error::Codec(
                    "prefill row: \"decode_allocation_policy\" is not a string".into(),
                ));
            }
        },
        enable_kv_cache_sharding: get_bool_default(v, "enable_kv_cache_sharding", false)?,
        prefill_http_port: get_opt_int(v, "prefill_http_port")?,
    })
}
