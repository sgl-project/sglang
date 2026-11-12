//! Scheduler primitives migrated from `python/sglang/srt/managers/schedule_policy.py`.
//!
//! This module ports waiting-queue ordering policies from Python. Cache
//! matching remains owned by the cache subsystem; the scheduler receives
//! matched prefix metadata and tree topology as explicit inputs.

/// One request's scheduler-visible waiting-queue fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitingRequest {
    /// Stable request identifier, used to return the selected request to its owner.
    pub id: String,
    /// Explicit priority; smaller or larger values may be configured as higher.
    pub priority: Option<i64>,
    /// Monotonic arrival sequence assigned when it enters the waiting queue.
    pub arrival_sequence: u64,
    /// Timestamp of waiting-queue entry, in monotonic microseconds.
    pub wait_started_micros: u64,
    /// Input tokens not already covered by the matched prefix.
    pub uncached_input_tokens: u64,
    /// Number of prefix tokens found in the KV cache.
    pub matched_prefix_tokens: u64,
    /// Output tokens already generated for this request.
    pub output_tokens: u64,
    /// Maximum output tokens requested.
    pub max_new_tokens: u64,
    /// Processed-prefill-token counter captured at queue entry.
    pub arrival_processed_tokens: u64,
    /// Whether prefix-cache policy temporarily deprioritized this request.
    pub temporarily_deprioritized: bool,
    /// Optional cache/routing affinity key.
    pub routing_key: Option<String>,
    /// Prefix-tree node index assigned by cache matching.
    pub prefix_node: usize,
}

/// Waiting-queue ordering mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitingOrder {
    /// Preserve arrival order.
    Fcfs,
    /// Sort by configured priority direction, then arrival order.
    PriorityFcfs { lower_value_first: bool },
    /// Highest response ratio next, aged by processed prefill tokens.
    Hrrn { processed_tokens: u64 },
    /// Longest cached prefix first, with temporary matches deferred.
    LongestPrefixMatch,
    /// Shortest uncached prefill first, with temporarily deprioritized requests last.
    ShortestPrefillFirst,
    /// Longest requested output first, optionally prefixed by priority order.
    LongestOutputFirst {
        priority_scheduling: bool,
        lower_value_first: bool,
    },
    /// Prefer keys that are already frequent in the active batch.
    RoutingKey {
        running_counts: Vec<(String, usize)>,
    },
    /// Depth-first order weighted by request count in each prefix subtree.
    DfsWeight { topology: PrefixTopology },
    /// Randomize the queue using a caller-supplied seed.
    Random { seed: u64 },
}

/// Order requests as the Python scheduler does for FCFS policies.
///
/// Priority mode requires every request to have a resolved priority. The
/// scheduler assigns the default priority on admission; returning an error
/// here catches a broken admission invariant instead of silently reordering.
pub fn order_waiting(
    requests: &mut [WaitingRequest],
    order: WaitingOrder,
) -> Result<(), ScheduleOrderError> {
    match order {
        WaitingOrder::Fcfs => requests.sort_by_key(|request| request.arrival_sequence),
        WaitingOrder::PriorityFcfs { lower_value_first } => {
            if requests.iter().any(|request| request.priority.is_none()) {
                return Err(ScheduleOrderError::MissingPriority);
            }
            requests.sort_by_key(|request| {
                let priority = request.priority.expect("validated above");
                let rank = if lower_value_first {
                    i128::from(priority)
                } else {
                    -i128::from(priority)
                };
                (rank, request.arrival_sequence)
            });
        }
        WaitingOrder::Hrrn { processed_tokens } => {
            requests.sort_by(|left, right| hrrn_cmp(left, right, processed_tokens))
        }
        WaitingOrder::LongestPrefixMatch => requests.sort_by_key(|request| {
            (
                request.temporarily_deprioritized,
                std::cmp::Reverse(request.matched_prefix_tokens),
            )
        }),
        WaitingOrder::ShortestPrefillFirst => requests.sort_by_key(|request| {
            (
                request.temporarily_deprioritized,
                request
                    .uncached_input_tokens
                    .saturating_add(request.output_tokens)
                    .max(1),
                request.wait_started_micros,
            )
        }),
        WaitingOrder::LongestOutputFirst {
            priority_scheduling,
            lower_value_first,
        } => {
            if priority_scheduling {
                if requests.iter().any(|request| request.priority.is_none()) {
                    return Err(ScheduleOrderError::MissingPriority);
                }
                requests.sort_by_key(|request| {
                    let priority = i128::from(request.priority.expect("validated above"));
                    let priority_rank = if lower_value_first {
                        priority
                    } else {
                        -priority
                    };
                    (priority_rank, std::cmp::Reverse(request.max_new_tokens))
                });
            } else {
                requests.sort_by_key(|request| std::cmp::Reverse(request.max_new_tokens));
            }
        }
        WaitingOrder::RoutingKey { running_counts } => {
            if !running_counts.is_empty() {
                let counts: std::collections::BTreeMap<_, _> = running_counts.into_iter().collect();
                requests.sort_by(|left, right| {
                    let left_key = left.routing_key.as_deref().unwrap_or("");
                    let right_key = right.routing_key.as_deref().unwrap_or("");
                    match (
                        left.routing_key.as_ref().and_then(|key| counts.get(key)),
                        right.routing_key.as_ref().and_then(|key| counts.get(key)),
                    ) {
                        (Some(left_count), Some(right_count)) => right_count
                            .cmp(left_count)
                            .then_with(|| left_key.cmp(right_key)),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => left_key.cmp(right_key),
                    }
                });
            }
        }
        WaitingOrder::DfsWeight { topology } => {
            let nodes: Vec<_> = requests.iter().map(|request| request.prefix_node).collect();
            let indices = topology
                .order(&nodes)
                .map_err(ScheduleOrderError::InvalidPrefixTopology)?;
            let original = requests.to_vec();
            for (target, source) in indices.into_iter().enumerate() {
                requests[target] = original[source].clone();
            }
        }
        WaitingOrder::Random { seed } => shuffle(requests, seed),
    }
    Ok(())
}

/// Prefix-tree topology snapshot used by DFS-weight ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixTopology {
    /// Child node indices in cache insertion order.
    pub children: Vec<Vec<usize>>,
    /// Root node index.
    pub root: usize,
}

impl PrefixTopology {
    /// Order request positions using subtree request counts as DFS weights.
    pub fn order(&self, request_nodes: &[usize]) -> Result<Vec<usize>, InvalidPrefixTopology> {
        let count = self.children.len();
        if self.root >= count || request_nodes.iter().any(|&node| node >= count) {
            return Err(InvalidPrefixTopology);
        }
        let mut at_node = vec![Vec::new(); count];
        for (index, &node) in request_nodes.iter().enumerate() {
            at_node[node].push(index);
        }
        let mut weights = vec![0_usize; count];
        let mut visited = vec![false; count];
        let mut stack = vec![(self.root, false)];
        while let Some((node, postorder)) = stack.pop() {
            if postorder {
                if std::mem::replace(&mut visited[node], true) {
                    return Err(InvalidPrefixTopology);
                }
                weights[node] = at_node[node].len();
                for &child in &self.children[node] {
                    weights[node] = weights[node]
                        .checked_add(weights[child])
                        .ok_or(InvalidPrefixTopology)?;
                }
            } else {
                if visited[node] || self.children[node].iter().any(|&child| child >= count) {
                    return Err(InvalidPrefixTopology);
                }
                stack.push((node, true));
                for &child in self.children[node].iter().rev() {
                    stack.push((child, false));
                }
            }
        }
        if visited.iter().any(|&seen| !seen) {
            return Err(InvalidPrefixTopology);
        }

        let mut order = Vec::with_capacity(request_nodes.len());
        let mut stack = vec![(self.root, false)];
        while let Some((node, postorder)) = stack.pop() {
            if postorder {
                order.extend(at_node[node].iter().copied());
            } else {
                stack.push((node, true));
                let mut children: Vec<_> = self.children[node]
                    .iter()
                    .copied()
                    .filter(|&child| weights[child] > 0)
                    .collect();
                children.sort_by_key(|&child| std::cmp::Reverse(weights[child]));
                stack.extend(children.into_iter().rev().map(|child| (child, false)));
            }
        }
        Ok(order)
    }
}

/// Invalid, cyclic, disconnected, or out-of-range prefix topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidPrefixTopology;

fn shuffle<T>(items: &mut [T], mut state: u64) {
    // A local seeded generator avoids process-global RNG state and makes the
    // decision reproducible across scheduler ranks.
    if items.len() < 2 {
        return;
    }
    if state == 0 {
        state = 0x9e37_79b9_7f4a_7c15;
    }
    for i in (1..items.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        items.swap(i, (state as usize) % (i + 1));
    }
}

fn hrrn_cmp(
    left: &WaitingRequest,
    right: &WaitingRequest,
    processed_tokens: u64,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (
        left.temporarily_deprioritized,
        right.temporarily_deprioritized,
    ) {
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        _ => {}
    }
    if left.temporarily_deprioritized {
        return left.id.cmp(&right.id);
    }

    let left_work = left.uncached_input_tokens;
    let right_work = right.uncached_input_tokens;
    match (left_work == 0, right_work == 0) {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        (true, true) => return left.id.cmp(&right.id),
        _ => {}
    }
    let left_wait = processed_tokens.saturating_sub(left.arrival_processed_tokens);
    let right_wait = processed_tokens.saturating_sub(right.arrival_processed_tokens);
    // Compare wait/work ratios without floating point or overflow. The common
    // `+1` in HRRN cancels when comparing ratios.
    let ordering =
        (right_wait as u128 * left_work as u128).cmp(&(left_wait as u128 * right_work as u128));
    ordering.then_with(|| left.id.cmp(&right.id))
}

/// Waiting-order policy could not be applied to the supplied request state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScheduleOrderError {
    /// A priority queue request reached ordering before admission resolved its priority.
    #[error("request priority was not resolved before scheduling")]
    MissingPriority,
    /// Prefix topology used by LPM/DFS ordering is invalid.
    #[error("prefix tree topology is invalid")]
    InvalidPrefixTopology(InvalidPrefixTopology),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: &str, priority: Option<i64>, arrival_sequence: u64) -> WaitingRequest {
        WaitingRequest {
            id: id.into(),
            priority,
            arrival_sequence,
            wait_started_micros: arrival_sequence,
            uncached_input_tokens: 1,
            matched_prefix_tokens: 0,
            output_tokens: 0,
            max_new_tokens: 1,
            arrival_processed_tokens: 0,
            temporarily_deprioritized: false,
            routing_key: None,
            prefix_node: 0,
        }
    }

    fn ids(requests: &[WaitingRequest]) -> Vec<&str> {
        requests.iter().map(|request| request.id.as_str()).collect()
    }

    #[test]
    fn fcfs_orders_by_arrival_sequence_not_storage_order() {
        let mut queue = vec![req("later", None, 8), req("first", None, 2)];
        order_waiting(&mut queue, WaitingOrder::Fcfs).unwrap();
        assert_eq!(ids(&queue), ["first", "later"]);
    }

    #[test]
    fn priority_fcfs_supports_both_priority_directions_and_stable_ties() {
        let input = vec![
            req("old-high", Some(1), 1),
            req("low", Some(5), 2),
            req("new-high", Some(1), 3),
        ];
        let mut lower_first = input.clone();
        order_waiting(
            &mut lower_first,
            WaitingOrder::PriorityFcfs {
                lower_value_first: true,
            },
        )
        .unwrap();
        assert_eq!(ids(&lower_first), ["old-high", "new-high", "low"]);

        let mut higher_first = input;
        order_waiting(
            &mut higher_first,
            WaitingOrder::PriorityFcfs {
                lower_value_first: false,
            },
        )
        .unwrap();
        assert_eq!(ids(&higher_first), ["low", "old-high", "new-high"]);
    }

    #[test]
    fn priority_mode_rejects_unresolved_priority_without_mutating_queue() {
        let mut queue = vec![req("a", Some(1), 1), req("b", None, 2)];
        let before = queue.clone();
        assert_eq!(
            order_waiting(
                &mut queue,
                WaitingOrder::PriorityFcfs {
                    lower_value_first: true,
                }
            ),
            Err(ScheduleOrderError::MissingPriority)
        );
        assert_eq!(queue, before);
    }

    #[test]
    fn hrrn_ages_by_processed_prefill_tokens_and_prioritizes_no_work() {
        let mut queue = vec![
            WaitingRequest {
                uncached_input_tokens: 100,
                arrival_processed_tokens: 100,
                ..req("young-expensive", None, 1)
            },
            WaitingRequest {
                uncached_input_tokens: 0,
                ..req("cached", None, 3)
            },
            WaitingRequest {
                uncached_input_tokens: 10,
                arrival_processed_tokens: 0,
                ..req("old-cheap", None, 2)
            },
        ];
        order_waiting(
            &mut queue,
            WaitingOrder::Hrrn {
                processed_tokens: 100,
            },
        )
        .unwrap();
        assert_eq!(ids(&queue), ["cached", "old-cheap", "young-expensive"]);
    }

    #[test]
    fn shortest_prefill_defers_temporary_prefix_matches() {
        let mut queue = vec![
            WaitingRequest {
                temporarily_deprioritized: true,
                uncached_input_tokens: 1,
                ..req("deferred", None, 1)
            },
            WaitingRequest {
                uncached_input_tokens: 20,
                ..req("long", None, 2)
            },
            WaitingRequest {
                uncached_input_tokens: 3,
                ..req("short", None, 3)
            },
        ];
        order_waiting(&mut queue, WaitingOrder::ShortestPrefillFirst).unwrap();
        assert_eq!(ids(&queue), ["short", "long", "deferred"]);
    }

    #[test]
    fn longest_output_first_respects_priority_then_output_length() {
        let mut queue = vec![
            WaitingRequest {
                max_new_tokens: 100,
                ..req("low", Some(5), 1)
            },
            WaitingRequest {
                max_new_tokens: 10,
                ..req("high-short", Some(1), 2)
            },
            WaitingRequest {
                max_new_tokens: 20,
                ..req("high-long", Some(1), 3)
            },
        ];
        order_waiting(
            &mut queue,
            WaitingOrder::LongestOutputFirst {
                priority_scheduling: true,
                lower_value_first: true,
            },
        )
        .unwrap();
        assert_eq!(ids(&queue), ["high-long", "high-short", "low"]);
    }

    #[test]
    fn longest_prefix_first_preserves_stable_ties_and_defers_temporary_matches() {
        let mut queue = vec![
            WaitingRequest {
                matched_prefix_tokens: 4,
                ..req("tie-a", None, 1)
            },
            WaitingRequest {
                matched_prefix_tokens: 4,
                ..req("tie-b", None, 2)
            },
            WaitingRequest {
                matched_prefix_tokens: 100,
                temporarily_deprioritized: true,
                ..req("deferred", None, 3)
            },
            WaitingRequest {
                matched_prefix_tokens: 8,
                ..req("best", None, 4)
            },
        ];
        order_waiting(&mut queue, WaitingOrder::LongestPrefixMatch).unwrap();
        assert_eq!(ids(&queue), ["best", "tie-a", "tie-b", "deferred"]);
    }

    #[test]
    fn routing_key_prefers_active_batch_frequency_and_keeps_stable_ties() {
        let mut first = req("x1", None, 1);
        first.routing_key = Some("x".into());
        let absent = req("none", None, 2);
        let mut second = req("y1", None, 3);
        second.routing_key = Some("y".into());
        let mut third = req("x2", None, 4);
        third.routing_key = Some("x".into());
        let mut queue = vec![first, absent, second, third];

        order_waiting(
            &mut queue,
            WaitingOrder::RoutingKey {
                running_counts: vec![("x".into(), 2), ("y".into(), 1)],
            },
        )
        .unwrap();
        assert_eq!(ids(&queue), ["x1", "x2", "y1", "none"]);
    }

    #[test]
    fn routing_key_without_running_keys_preserves_queue_order() {
        let mut keyed = req("keyed", None, 1);
        keyed.routing_key = Some("k".into());
        let mut queue = vec![keyed, req("other", None, 2)];
        order_waiting(
            &mut queue,
            WaitingOrder::RoutingKey {
                running_counts: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(ids(&queue), ["keyed", "other"]);
    }

    #[test]
    fn random_policy_is_seeded_and_permutation_preserving() {
        let original = vec![
            req("a", None, 1),
            req("b", None, 2),
            req("c", None, 3),
            req("d", None, 4),
        ];
        let mut first = original.clone();
        let mut repeat = original.clone();
        let mut different = original.clone();
        order_waiting(&mut first, WaitingOrder::Random { seed: 19 }).unwrap();
        order_waiting(&mut repeat, WaitingOrder::Random { seed: 19 }).unwrap();
        order_waiting(&mut different, WaitingOrder::Random { seed: 47 }).unwrap();
        assert_eq!(first, repeat);
        assert_ne!(ids(&first), ids(&original));
        assert_ne!(ids(&first), ids(&different));
        let mut result = ids(&first);
        result.sort_unstable();
        assert_eq!(result, ["a", "b", "c", "d"]);
    }

    #[test]
    fn dfs_weight_visits_heaviest_subtree_first_and_keeps_duplicate_nodes_stable() {
        let topology = PrefixTopology {
            root: 0,
            children: vec![vec![1, 2], vec![3], vec![], vec![]],
        };
        let nodes = [3, 3, 2, 1];
        assert_eq!(topology.order(&nodes).unwrap(), [0, 1, 3, 2]);
    }

    #[test]
    fn dfs_weight_rejects_malformed_or_disconnected_topology() {
        let bad_child = PrefixTopology {
            root: 0,
            children: vec![vec![1]],
        };
        assert_eq!(bad_child.order(&[0]), Err(InvalidPrefixTopology));
        let disconnected = PrefixTopology {
            root: 0,
            children: vec![vec![], vec![]],
        };
        assert_eq!(disconnected.order(&[0]), Err(InvalidPrefixTopology));
    }
}
