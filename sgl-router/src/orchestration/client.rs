//! Bounded Engine HTTP adapter. Control responses never become residency evidence.
use bytes::BytesMut;
use futures::StreamExt;
use reqwest::{header::HeaderValue, Client};
use serde_json::{json, Value};
use sgl_kv_orchestrator::models::{Compatibility, PrefixTarget};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Capabilities {
    pub engine_id: String,
    pub epoch: String,
    pub compatibility: Compatibility,
    pub public: Value,
    pub generation: Option<u64>,
    pub feedback: Option<Value>,
}
impl Capabilities {
    pub fn parse(value: Value) -> Result<Self, String> {
        let public = value["public_protocol"].clone();
        let p = &public["prefix"];
        let compatibility = Compatibility {
            namespace: value["namespace"]
                .as_str()
                .ok_or("namespace absent")?
                .into(),
            cache_salt: None,
            layout_id: p["layout_id"].as_str().ok_or("layout absent")?.into(),
            page_size: p["page_size"]
                .as_u64()
                .filter(|n| (1..=256).contains(n))
                .ok_or("page size absent")?,
            components: serde_json::from_value(p["components"].clone())
                .map_err(|_| "components absent")?,
            hash_scheme: p["hash_scheme"]
                .as_str()
                .ok_or("hash scheme absent")?
                .into(),
        };
        sgl_kv_orchestrator::session_index::ContentKey::new(
            compatibility.clone(),
            "validate".into(),
        )
        .map_err(|_| "invalid compatibility")?;
        Ok(Self {
            engine_id: value["engine_id"]
                .as_str()
                .ok_or("engine id absent")?
                .into(),
            epoch: value["engine_epoch"]
                .as_str()
                .or_else(|| value["epoch"].as_str())
                .ok_or("epoch absent")?
                .into(),
            generation: value["executor"]["cache_generation"].as_u64(),
            feedback: value.get("feedback").filter(|v| !v.is_null()).cloned(),
            compatibility,
            public,
        })
    }
    pub fn context(&self, caller: &str) -> Value {
        json!({"engine_id":self.engine_id,"engine_epoch":self.epoch,"namespace":self.compatibility.namespace,"caller_id":caller,"binding_id":null})
    }
    pub fn compatible(&self, prefix: &PrefixTarget) -> bool {
        let mut expected = self.compatibility.clone();
        expected.cache_salt = prefix.compatibility.cache_salt.clone();
        let mut actual = prefix.compatibility.clone();
        actual.components.sort();
        expected.components.sort();
        actual == expected
    }
    pub fn path(&self, action: &str, source: &str, destination: &str) -> bool {
        self.public["actions"][action]
            .as_array()
            .is_some_and(|a| a.iter().any(|v| v == "1.0"))
            && self.public["paths"][action]
                .as_array()
                .is_some_and(|paths| paths.iter().any(|v| v == &json!([source, destination])))
    }
}
#[derive(Clone)]
pub struct EngineClient {
    client: Client,
    max_bytes: usize,
    auth: Option<HeaderValue>,
}
impl EngineClient {
    pub fn new(timeout_ms: u64, max_bytes: usize, auth: Option<HeaderValue>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_millis(timeout_ms))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("rustls client"),
            max_bytes,
            auth,
        }
    }
    async fn request(&self, url: &str, body: Option<&Value>) -> Result<Value, String> {
        let mut req = if let Some(body) = body {
            self.client.post(format!("{url}/v1/kv/control")).json(body)
        } else {
            self.client.get(format!("{url}/v1/kv/capabilities"))
        };
        if let Some(auth) = &self.auth {
            req = req.header("authorization", auth);
        }
        let response = req.send().await.map_err(|e| e.to_string())?;
        let status = response.status();
        if response
            .content_length()
            .is_some_and(|n| n > self.max_bytes as u64)
        {
            return Err("engine response capacity".into());
        }
        let mut stream = response.bytes_stream();
        let mut bytes = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            if chunk.len() > self.max_bytes.saturating_sub(bytes.len()) {
                return Err("engine response capacity".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if !status.is_success() || value["success"] != true {
            return Err(format!("engine {status}: {}", value["error"]));
        }
        Ok(value["data"].clone())
    }
    pub async fn capabilities(&self, url: &str) -> Result<Capabilities, String> {
        Capabilities::parse(self.request(url, None).await?)
    }
    pub async fn command(&self, url: &str, body: &Value) -> Result<Value, String> {
        self.request(url, Some(body)).await
    }
    /// Logical TokensTarget is understood by residency; public PrefixTarget is
    /// normalized only on submit. No Engine-local session binding is required.
    pub async fn residency(
        &self,
        url: &str,
        caps: &Capabilities,
        caller: &str,
        prefix: &PrefixTarget,
        tier: &str,
    ) -> Result<Value, String> {
        self.command(url,&json!({"kind":"residency","context":caps.context(caller),"limit":256,
            "target":{"kind":"tokens","namespace":prefix.compatibility.namespace,"token_ids":prefix.token_ids,
                "cache_salt":prefix.compatibility.cache_salt,"layout_id":prefix.compatibility.layout_id},
            "location":{"tier":tier,"engine_id":caps.engine_id}})).await
    }
}

/// Returns continuous page-aligned FULL-component coverage proven by this query.
/// Missing or unsupported/window/checkpoint components never imply full readiness.
pub fn confirmed_end(
    snapshot: &Value,
    caps: &Capabilities,
    prefix: &PrefixTarget,
    tier: &str,
) -> Result<u64, String> {
    if snapshot["engine_epoch"] != caps.epoch
        || snapshot["namespace"] != caps.compatibility.namespace
        || caps
            .generation
            .is_some_and(|g| snapshot["cache_generation"].as_u64() != Some(g))
    {
        return Err("stale residency owner/generation".into());
    }
    let items = snapshot["items"]
        .as_array()
        .ok_or("residency items absent")?;
    let mut end = prefix.end_token;
    for component in &prefix.compatibility.components {
        let mut spans: Vec<(u64, u64)> = items
            .iter()
            .filter_map(|row| {
                let d = &row["descriptor"];
                if row["coverage_kind"] != "tokens"
                    || d["component"] != *component
                    || row["location"]["tier"] != tier
                    || row["location"]["engine_id"] != caps.engine_id
                {
                    return None;
                }
                let start = d["start_token"].as_u64()?;
                let finish = d["end_token"].as_u64()?;
                if start >= finish
                    || finish > prefix.end_token
                    || start % prefix.compatibility.page_size != 0
                    || finish % prefix.compatibility.page_size != 0
                    || d["dependency_start"].as_u64().unwrap_or(0) > start
                {
                    return None;
                }
                Some((start, finish))
            })
            .collect();
        spans.sort();
        let mut covered = 0;
        for (start, finish) in spans {
            if start > covered {
                break;
            }
            covered = covered.max(finish);
        }
        end = end.min(covered);
    }
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_engine_owns_salt_transformation() {
        let value = json!({"engine_id":"worker-a","engine_epoch":"epoch","namespace":"weights",
            "cache_salt_encoding":r#"sha256 of compact UTF-8 JSON ["sglang.kv.weight.v1", namespace, original_salt_or_null]"#,
            "public_protocol":{"prefix":{"layout_id":"full:p64","page_size":64,"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"}}});
        let caps = Capabilities::parse(value).unwrap();
        let mut c = caps.compatibility.clone();
        c.cache_salt = Some("raw-trial-salt".into());
        let prefix = PrefixTarget {
            kind: "prefix".into(),
            compatibility: c,
            start_token: 0,
            end_token: 64,
            token_ids: Some(vec![1; 64]),
            blocks: vec![],
        };
        assert!(caps.compatible(&prefix));
        assert_eq!(
            prefix.compatibility.cache_salt.as_deref(),
            Some("raw-trial-salt")
        );
    }
}
