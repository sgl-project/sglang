use std::collections::HashSet;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{HeaderMap, StatusCode};
use futures::future::join_all;
use serde_json::Value;

use super::DpState;

#[derive(Debug)]
pub(super) struct RouteError {
    pub status: StatusCode,
    pub message: String,
}

impl RouteError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "selected DP workers are unavailable".into(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum LoadBalanceMethod {
    RoundRobin,
    FollowBootstrapRoom,
    TotalRequests,
    TotalTokens,
}

impl FromStr for LoadBalanceMethod {
    type Err = String;

    fn from_str(method: &str) -> Result<Self, Self::Err> {
        match method.to_ascii_lowercase().as_str() {
            "round_robin" => Ok(Self::RoundRobin),
            "follow_bootstrap_room" => Ok(Self::FollowBootstrapRoom),
            "total_requests" => Ok(Self::TotalRequests),
            "total_tokens" => Ok(Self::TotalTokens),
            _ => Err(format!("invalid load balance method: {method}")),
        }
    }
}

#[derive(Default)]
struct Budget {
    timestamp: f64,
    requests: u64,
    tokens: u64,
}

pub(super) struct RoutingState {
    method: LoadBalanceMethod,
    next: usize,
    budgets: Vec<Budget>,
    available: Vec<bool>,
}

impl RoutingState {
    pub(super) fn new(method: LoadBalanceMethod, size: usize) -> Self {
        Self {
            method,
            next: 0,
            budgets: (0..size).map(|_| Budget::default()).collect(),
            available: vec![true; size],
        }
    }

    fn choose(
        &mut self,
        rank: Option<i64>,
        room: Option<i64>,
        tokens: u64,
    ) -> Result<usize, RouteError> {
        let size = self.budgets.len();
        if let Some(rank) = rank {
            let rank = usize::try_from(rank)
                .ok()
                .filter(|rank| *rank < size)
                .ok_or_else(|| {
                    RouteError::invalid(format!("routed_dp_rank must be in [0, {size})"))
                })?;
            return self.available[rank]
                .then_some(rank)
                .ok_or_else(RouteError::unavailable);
        }
        let rank = match self.method {
            LoadBalanceMethod::RoundRobin => {
                let rank = (0..size)
                    .map(|offset| (self.next + offset) % size)
                    .find(|rank| self.available[*rank])
                    .ok_or_else(RouteError::unavailable)?;
                self.next = (rank + 1) % size;
                rank
            }
            LoadBalanceMethod::FollowBootstrapRoom => {
                room.ok_or_else(|| {
                    RouteError::invalid(
                        "bootstrap_room is required for follow_bootstrap_room routing",
                    )
                })?
                .rem_euclid(size as i64) as usize
            }
            LoadBalanceMethod::TotalRequests | LoadBalanceMethod::TotalTokens => self
                .budgets
                .iter()
                .enumerate()
                .filter(|(rank, _)| self.available[*rank])
                .min_by_key(|(_, budget)| match self.method {
                    LoadBalanceMethod::TotalTokens => (budget.tokens, budget.requests),
                    _ => (budget.requests, 0),
                })
                .map(|(rank, _)| rank)
                .ok_or_else(RouteError::unavailable)?,
        };
        if !self.available[rank] {
            return Err(RouteError::unavailable());
        }
        let budget = &mut self.budgets[rank];
        budget.requests = budget.requests.saturating_add(1);
        budget.tokens = budget.tokens.saturating_add(tokens);
        Ok(rank)
    }

    fn update(&mut self, load: &Value) {
        let Some(rank) = load["dp_rank"].as_u64().map(|rank| rank as usize) else {
            return;
        };
        let Some(budget) = self.budgets.get_mut(rank) else {
            return;
        };
        let Some(timestamp) = load["timestamp"].as_f64() else {
            return;
        };
        if timestamp != budget.timestamp {
            budget.timestamp = timestamp;
            budget.requests = load["num_running_reqs"]
                .as_u64()
                .unwrap_or(0)
                .saturating_add(load["num_waiting_reqs"].as_u64().unwrap_or(0));
            budget.tokens = load["num_total_tokens"].as_u64().unwrap_or(0);
        }
    }
}

pub(super) fn choose(
    state: &DpState,
    rank: Option<i64>,
    room: Option<i64>,
    tokens: u64,
) -> Result<usize, RouteError> {
    state
        .routing
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .choose(rank, room, tokens)
}

pub(super) fn rank_hint(
    value: &Value,
    headers: &HeaderMap,
    overrides: bool,
) -> Result<Option<i64>, String> {
    if overrides
        && let Some(rank) =
            crate::api_server::headers::integer(headers, "x-override-routed-dp-rank")?
    {
        return Ok(Some(rank));
    }
    let value = value
        .get("routed_dp_rank")
        .filter(|value| !value.is_null())
        .or_else(|| {
            value
                .get("data_parallel_rank")
                .filter(|value| !value.is_null())
        });
    value
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| "routed_dp_rank must be an integer".to_owned())
        })
        .transpose()
}

pub(super) fn validate_workers(
    mut workers: Vec<(usize, String)>,
    public: SocketAddr,
) -> Result<Vec<String>, String> {
    if workers.is_empty() {
        return Err("DP ingress needs at least one worker".into());
    }
    workers.sort_by_key(|(rank, _)| *rank);
    let mut urls = HashSet::new();
    workers
        .into_iter()
        .enumerate()
        .map(|(expected, (rank, url))| {
            if rank != expected {
                return Err("DP worker ranks must cover [0, dp_size) exactly once".into());
            }
            let parsed = reqwest::Url::parse(&url).map_err(|e| e.to_string())?;
            if parsed.scheme() != "http"
                || parsed.host_str().is_none()
                || !parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.query().is_some()
                || parsed.fragment().is_some()
                || parsed.path() != "/"
            {
                return Err("DP worker URL must be an HTTP origin".into());
            }
            if parsed.port_or_known_default() == Some(public.port())
                && parsed.host_str().is_some_and(|host| {
                    host.trim_matches(['[', ']'])
                        .parse::<std::net::IpAddr>()
                        .ok()
                        == Some(public.ip())
                })
            {
                return Err("DP worker URL points to the public listener".into());
            }
            let url = parsed.as_str().trim_end_matches('/').to_owned();
            if !urls.insert(url.clone()) {
                return Err("DP worker URLs must be distinct".into());
            }
            Ok(url)
        })
        .collect()
}

pub(super) async fn refresh_loads(state: Arc<DpState>) {
    loop {
        let state = &state;
        join_all(
            state
                .workers
                .iter()
                .enumerate()
                .map(|(rank, worker)| async move {
                    let result = async {
                        let response = state
                            .client
                            .get(format!("{worker}/v1/loads?include=core"))
                            .timeout(Duration::from_secs(1))
                            .send()
                            .await?
                            .error_for_status()?;
                        let bytes = response.bytes().await?;
                        Ok::<_, reqwest::Error>(bytes)
                    }
                    .await;
                    let value = result
                        .ok()
                        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
                    let load = value
                        .as_ref()
                        .and_then(|value| value["loads"].as_array())
                        .filter(|loads| loads.len() == 1)
                        .and_then(|loads| loads.first())
                        .filter(|load| load["dp_rank"].as_u64() == Some(rank as u64));
                    let mut routing = state
                        .routing
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if routing.available[rank] != load.is_some() {
                        tracing::warn!(
                            rank,
                            available = load.is_some(),
                            "DP worker load availability changed"
                        );
                    }
                    routing.available[rank] = load.is_some();
                    if let Some(load) = load {
                        routing.update(load);
                    }
                }),
        )
        .await;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn selection_respects_loads_stale_snapshots_affinity_and_unavailable_workers() {
        for method in [
            LoadBalanceMethod::TotalRequests,
            LoadBalanceMethod::TotalTokens,
        ] {
            let mut routing = RoutingState::new(method, 2);
            let load = json!({"dp_rank":0,"timestamp":1.0,"num_running_reqs":0,"num_waiting_reqs":0,"num_total_tokens":0});
            routing.update(&load);
            assert_eq!(routing.choose(None, None, 5).unwrap(), 0);
            routing.update(&load);
            assert_eq!(routing.choose(None, None, 5).unwrap(), 1);
            assert_eq!(routing.choose(None, None, 5).unwrap(), 0);
            routing.available[1] = false;
            assert_eq!(routing.choose(None, None, 5).unwrap(), 0);
            assert_eq!(
                routing.choose(Some(1), None, 0).unwrap_err().status,
                StatusCode::SERVICE_UNAVAILABLE
            );
            assert_eq!(
                routing.choose(Some(2), None, 0).unwrap_err().status,
                StatusCode::BAD_REQUEST
            );
        }
        let mut routing = RoutingState::new(LoadBalanceMethod::FollowBootstrapRoom, 2);
        assert_eq!(routing.choose(None, Some(13), 0).unwrap(), 1);
        assert_eq!(routing.choose(Some(0), Some(13), 0).unwrap(), 0);
        assert!(routing.choose(None, None, 0).is_err());
        routing.available[1] = false;
        assert_eq!(
            routing.choose(None, Some(13), 0).unwrap_err().status,
            StatusCode::SERVICE_UNAVAILABLE
        );

        for workers in [
            vec![],
            vec![(1, "http://127.0.0.1:1".into())],
            vec![
                (0, "http://127.0.0.1:1".into()),
                (1, "http://127.0.0.1:1".into()),
            ],
        ] {
            assert!(validate_workers(workers, "127.0.0.1:2".parse().unwrap()).is_err());
        }
    }
}
