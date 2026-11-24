// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{BlockSizeOracle, CacheNamespace, HashTree};
use crate::tokenizer::MEDIA_TAG_LEN;
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Prefix lookup result from the local radix tree or remote KV indexer,
/// consumed by cache-aware routing.
pub struct PrefixLookupResult {
    pub outcome: PrefixOutcome,
    pub query_blocks: usize,
    /// Workers whose reported depth came from a pending stake rather than
    /// from KV events (empty when the confirmed tree decided the lookup).
    /// Routing to one of them is a "follow" of that stake.
    pub pending_winners: Vec<String>,
    /// The query's block hashes when the local tree answered, so routing can
    /// record the placement without rehashing.
    pub block_hashes: Option<Arc<[i64]>>,
    /// The query's media chain (see `RadixTreePrefixProvider::media_chain`),
    /// from its conversation key on, when it is a media chat under media
    /// affinity, so routing can record where the chat went.
    pub media_hashes: Option<Arc<[i64]>>,
    /// Workers whose reported depth is a remembered media placement: deeper
    /// than every confirmed match, and at least as deep as any other signal
    /// for that worker. The cache gate's token floor does not apply
    /// to them: it counts routing tokens, and the image KV and vision encoding
    /// such a placement stands for are not among them.
    pub media_winners: Vec<String>,
}

/// A worker's DP ranks holding a prompt (see [`RadixTreePrefixProvider::rank_depths`]).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RankDepths {
    /// `(dp_rank, cached prefix blocks)`, from KV events and, under media
    /// affinity, the rank remembered for the chat.
    pub depths: Vec<(u32, usize)>,
    /// Under media affinity with no rank remembered for the chat: the rank a
    /// rendezvous hash of the conversation prefers among those tied at the
    /// deepest depth, so every replica prefers the same one. A tie-break only:
    /// rank load decides first (see `select_dp_rank`).
    pub preferred: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct RadixTreePrefixProvider {
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
    /// Confirmed-first mode, on when set: a pending stake is consulted only
    /// when it is strictly deeper than the best confirmed KV-event match, and
    /// only while the confirmed match covers less than this share of the
    /// prompt. Above it the confirmed tree already routes the prompt and its
    /// successor turns, so the pending walk and the stake write are skipped:
    /// a warm request pays nothing for a tier it cannot benefit from. `None`
    /// keeps the original behavior: pending depths are merged with the
    /// confirmed ones by maximum on every request, and every request stakes.
    pending_consult_below: Option<f64>,
}

impl RadixTreePrefixProvider {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>) -> Self {
        Self {
            tree,
            block_size_oracle,
            pending_consult_below: None,
        }
    }

    /// Switch on confirmed-first mode: consult and stake pending prefixes
    /// only when the confirmed match covers less than `ratio` of the prompt
    /// and only where a stake is strictly deeper (see `pending_consult_below`).
    pub fn with_pending_consult_below(mut self, ratio: f64) -> Self {
        self.pending_consult_below = Some(ratio);
        self
    }

    #[cfg(test)]
    pub(crate) fn pending_consult_below(&self) -> Option<f64> {
        self.pending_consult_below
    }

    pub fn match_request_tokens(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
    ) -> Option<PrefixLookupResult> {
        let hashes = self.block_hashes(tokens, namespace)?;

        let mut depth_by_url = BTreeMap::<String, u32>::new();
        for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
            merge_depth(&mut depth_by_url, worker.url, depth);
        }
        let confirmed = depth_by_url.values().copied().max();
        let confirmed_by_url = depth_by_url.clone();
        // Stakes at their follower cap are already filtered out by
        // `PendingPrefixes::depths`.
        let (consult_pending, merge_pending) = match self.pending_consult_below {
            // Original behavior: walk the pending store on every request and
            // merge its depths with the confirmed ones by maximum.
            None => (self.tree.pending().is_enabled(), true),
            // Confirmed-first: pay for the pending tier only when the confirmed
            // tree has not already placed this prompt (a first turn, or a turn
            // racing its predecessor's KV events), and let a stake count only
            // where it is strictly deeper than the best confirmed match, so a
            // real match that is at least as good always decides.
            Some(below) => {
                let covered = confirmed.unwrap_or(0) as f64 / hashes.len() as f64;
                (self.tree.pending().is_enabled() && covered < below, false)
            }
        };
        let pending = if consult_pending {
            self.tree.pending().depths(&hashes)
        } else {
            Vec::new()
        };
        // Confirmed-first merges a stake only where it is strictly deeper than
        // the best confirmed match, each on its own: a stake no deeper than
        // that match never becomes a candidate, even beside one that is.
        for (url, depth) in pending {
            let deeper = Some(u32::try_from(depth).unwrap_or(u32::MAX)) > confirmed;
            if merge_pending || deeper {
                merge_depth(&mut depth_by_url, url.to_string(), depth);
            }
        }
        // Taken before the media tier: only a worker that won through its pending
        // stake is a follower of that stake, not one a remembered placement or
        // the rendezvous tie-break below prefers.
        let pending_winners: Vec<String> = depth_by_url
            .iter()
            .filter(|(url, depth)| confirmed_by_url.get(*url) < Some(depth))
            .map(|(url, _)| url.clone())
            .collect();
        // A chat with images is confirmed by KV events only up to its first image,
        // so a placement remembered past that point counts where it is strictly
        // deeper than the best confirmed match. A real deeper hit still decides.
        let pending_best = depth_by_url.values().copied().max();
        let media = self.media_chain(tokens, namespace, &hashes);
        let mut media_winners = Vec::new();
        if let Some(media) = &media {
            let mut deepened = false;
            for (url, matched) in self.tree.media().depths(&media.chain) {
                let depth = media.depth(matched);
                let depth_blocks = u32::try_from(depth).unwrap_or(u32::MAX);
                if Some(depth_blocks) > confirmed {
                    if depth_by_url
                        .get(&*url)
                        .is_none_or(|known| depth_blocks >= *known)
                    {
                        media_winners.push(url.to_string());
                    }
                    merge_depth(&mut depth_by_url, url.to_string(), depth);
                    deepened = true;
                }
            }
            if deepened {
                self.tree.media().record_hit();
            } else if prefer_rendezvous(&mut depth_by_url, media.key) {
                // No placement remembered here (another router may have routed the
                // previous turn): every router breaks the tie on the shared prefix
                // the same way for the same conversation, by rendezvous hash.
                self.tree.record_media_tiebreak();
            }
        }
        let outcome = match depth_by_url.values().copied().max() {
            None => PrefixOutcome::Empty,
            Some(best_prefix_blocks) => {
                if confirmed < pending_best {
                    self.tree.pending().record_hit();
                }
                let matches = depth_by_url
                    .into_iter()
                    .map(|(address, matched_prefix_blocks)| PrefixMatch {
                        worker_id: address.clone(),
                        address,
                        matched_prefix_blocks,
                    })
                    .collect();
                PrefixOutcome::Matched {
                    matches,
                    best_prefix_blocks,
                }
            }
        };
        // Empty is still returned so routing can record where this prompt went.
        Some(PrefixLookupResult {
            outcome,
            query_blocks: hashes.len(),
            pending_winners,
            media_hashes: media.map(|media| Arc::<[i64]>::from(media.chain)),
            media_winners,
            block_hashes: consult_pending.then(|| hashes.into()),
        })
    }

    /// Remember that `signal`'s prompt was just routed to `url`. When the
    /// placement followed `url`'s pending stake, charge that stake a follower
    /// first, so the cap sees every request the stake attracted.
    pub fn record_route(&self, signal: &PrefixLookupResult, url: &str) {
        if let Some(hashes) = &signal.media_hashes {
            self.tree.media().record(url, hashes);
        }
        if let Some(hashes) = &signal.block_hashes {
            if signal.pending_winners.iter().any(|w| w == url) {
                self.tree.pending().record_follow(url, hashes);
            }
            self.tree.pending().record(url, hashes);
        }
    }

    /// The ranks of `worker_url` holding the prompt (see [`RankDepths`]).
    pub fn rank_depths(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
        worker_url: &str,
    ) -> RankDepths {
        let Some(hashes) = self.block_hashes(tokens, namespace) else {
            return RankDepths::default();
        };
        let mut depths: Vec<(u32, usize)> = self
            .tree
            .prefix_depths(None, &hashes)
            .into_iter()
            .filter(|(worker, _)| worker.url == worker_url)
            .map(|(worker, depth)| (worker.dp_rank, depth))
            .collect();
        let mut preferred = None;
        if let Some(media) = self.media_chain(tokens, namespace, &hashes) {
            let before = depths.iter().map(|&(_, d)| d).max();
            let owner_prefix = format!("{worker_url}#");
            let mut remembered = false;
            for (owner, matched) in self.tree.media_ranks().depths(&media.chain) {
                let depth = media.depth(matched);
                // Confirmed first, as for workers: a remembered rank counts only
                // where it is strictly deeper than every confirmed match here, so
                // a rank known to hold the prefix is never outranked by a stale
                // or unfinished placement that holds no more of it.
                if Some(depth) <= before {
                    continue;
                }
                let Some(rank) = owner
                    .strip_prefix(owner_prefix.as_str())
                    .and_then(|rank| rank.parse::<u32>().ok())
                else {
                    continue;
                };
                match depths.iter_mut().find(|(r, _)| *r == rank) {
                    Some((_, d)) => *d = (*d).max(depth),
                    None => depths.push((rank, depth)),
                }
                remembered = true;
            }
            if !remembered {
                // No rank remembered (another router may have routed the previous
                // turn): every router prefers the same tied rank, by rendezvous
                // hash. A preference only: rank load decides first, as nothing at
                // rank level guards against a loaded rank the way the pressure
                // guard does for workers. Not counted as a tie-break: the
                // selector may not follow it, and the request's worker-level
                // lookup already counted the conversation once.
                let by_rank: BTreeMap<String, u32> = depths
                    .iter()
                    .map(|&(rank, d)| (rank.to_string(), u32::try_from(d).unwrap_or(u32::MAX)))
                    .collect();
                if let Some(winner) = rendezvous_winner(&by_rank, media.key) {
                    preferred = winner.parse().ok();
                }
            }
        }
        RankDepths { depths, preferred }
    }

    /// A media chat's routing chain; `None` for a text prompt or with media
    /// affinity off.
    ///
    /// The chain starts at the conversation key: the hash of the tokens of the
    /// block holding the first image, through the image and its identity tag
    /// (see [`crate::tokenizer::tag_images`]), chained to the block before it.
    /// The prompt's block hashes after that block follow. The text after the
    /// image in its block, and so the block's own hash, changes as the
    /// conversation grows (a first turn often ends in it); the key does not.
    /// So every turn of the chat finds the placement its first turn recorded,
    /// and every router breaks the same tie the same way, while a later turn
    /// still matches a longer placement block by block.
    ///
    /// The text before the image is not in the chain: the key covers it
    /// through its parent hash, and a placement counts only once it matches
    /// the key, so a chat that shares only that text with another chat (a
    /// burst of new chats on one system prompt) inherits nothing from it. KV
    /// events never confirm the image's block or any after it (the engine's
    /// keys carry the image's own hash), so the confirmed match, at most the
    /// blocks before it, is always shallower than a matching placement.
    fn media_chain(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
        hashes: &[i64],
    ) -> Option<MediaChain> {
        let position = self.tree.first_media_position(tokens)?;
        let block_size = usize::try_from(self.block_size_oracle.get()?).ok()?.max(1);
        let block = position / block_size;
        let end = (position + 1 + MEDIA_TAG_LEN).min(tokens.len());
        let parent = block.checked_sub(1).and_then(|i| hashes.get(i).copied());
        let key = media_key(namespace, parent, &tokens[block * block_size..end]);
        let mut chain = Vec::with_capacity(hashes.len().saturating_sub(block));
        chain.push(key);
        chain.extend(hashes.iter().skip(block + 1));
        Some(MediaChain { chain, key, block })
    }

    /// Remember the DP rank a media chat was placed on, so its next turn goes
    /// back to that rank's cache (see [`HashTree::media_ranks`]).
    pub fn record_rank(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
        worker_url: &str,
        rank: u32,
    ) {
        // A text prompt pays no hashing here (the match already hashed it).
        if !self.tree.is_media_prompt(tokens) {
            return;
        }
        let Some(hashes) = self.block_hashes(tokens, namespace) else {
            return;
        };
        if let Some(media) = self.media_chain(tokens, namespace, &hashes) {
            self.tree
                .media_ranks()
                .record(&format!("{worker_url}#{rank}"), &media.chain);
        }
    }

    fn block_hashes(&self, tokens: &[u32], namespace: &CacheNamespace) -> Option<Vec<i64>> {
        let hashes = self.block_size_oracle.block_hashes(tokens, namespace)?;
        (!hashes.is_empty()).then_some(hashes)
    }
}

/// A media chat's routing chain (see [`RadixTreePrefixProvider::media_chain`]).
struct MediaChain {
    /// The conversation key, then the block hashes after the first image's block.
    chain: Vec<i64>,
    key: i64,
    /// Index of the first image's block in the prompt's block hashes.
    block: usize,
}

impl MediaChain {
    /// The prompt depth, in blocks, of a placement matching `matched` entries
    /// of the chain: the blocks before the image's, then the matched ones.
    fn depth(&self, matched: usize) -> usize {
        self.block + matched
    }
}

/// A media chat's conversation key (see [`RadixTreePrefixProvider::media_chain`]):
/// router-only, so it follows no engine's hash layout, only the namespace that
/// keeps chats with a cache salt or a LoRA apart, and is the same on every router.
fn media_key(namespace: &CacheNamespace, parent: Option<i64>, tokens: &[u32]) -> i64 {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"sgl-router-media-key-v1\0");
    for part in [&namespace.cache_salt, &namespace.lora_name] {
        hasher.update(part.as_deref().unwrap_or("").as_bytes());
        hasher.update([0u8]);
    }
    if let Some(parent) = parent {
        hasher.update(parent.to_be_bytes());
    }
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    super::sha256_to_i64(&hasher.finalize().into())
}

/// Among the entries tied at the deepest depth, credit the one with the highest
/// rendezvous score for `key` with one more block, so a cache-aware pick prefers
/// it. Deterministic across routers; ties of one entry are left alone.
fn prefer_rendezvous(depths: &mut BTreeMap<String, u32>, key: i64) -> bool {
    if let Some(winner) = rendezvous_winner(depths, key) {
        let deepest = depths[&winner];
        depths.insert(winner, deepest.saturating_add(1));
        return true;
    }
    false
}

/// Among the names tied at the deepest depth (two or more, above zero), the one
/// a rendezvous hash of `key` ranks first; `None` when there is no such tie.
fn rendezvous_winner(depths: &BTreeMap<String, u32>, key: i64) -> Option<String> {
    let deepest = *depths.values().max()?;
    if deepest == 0 || depths.values().filter(|&&d| d == deepest).count() < 2 {
        return None;
    }
    let score = |name: &str| {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&(key, name), &mut hasher);
        std::hash::Hasher::finish(&hasher)
    };
    depths
        .iter()
        .filter(|(_, &d)| d == deepest)
        .max_by_key(|(name, _)| score(name))
        .map(|(name, _)| name.clone())
}

fn merge_depth(depth_by_url: &mut BTreeMap<String, u32>, url: String, depth: usize) {
    let depth = u32::try_from(depth).unwrap_or(u32::MAX);
    depth_by_url
        .entry(url)
        .and_modify(|current| *current = (*current).max(depth))
        .or_insert(depth);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::kv_events::tree::KvWorkerId;
    use crate::state::kv_events::wire::BlockStored;
    use crate::tokenizer::MEDIA_TAG_MARK;
    use std::time::Duration;

    const BLOCK: u32 = 2;

    fn provider(ttl_ms: u64, max_followers: u64) -> RadixTreePrefixProvider {
        let tree = Arc::new(HashTree::new());
        let oracle = BlockSizeOracle::new();
        oracle.try_set(BLOCK).expect("block size");
        tree.pending().enable(Duration::from_millis(ttl_ms));
        tree.pending().set_max_followers(max_followers);
        RadixTreePrefixProvider::new(tree, oracle)
    }

    fn tokens(blocks: usize) -> Vec<u32> {
        (1..=(blocks * BLOCK as usize) as u32).collect()
    }

    /// Publish the first `blocks` blocks of `tokens` for `url` as KV events.
    fn confirm(provider: &RadixTreePrefixProvider, url: &str, tokens: &[u32], blocks: usize) {
        let hashes = provider
            .block_hashes(tokens, &CacheNamespace::default())
            .expect("hashes");
        let event = BlockStored {
            block_hashes: hashes[..blocks].to_vec(),
            parent_block_hash: None,
            token_ids: tokens[..blocks * BLOCK as usize].to_vec(),
            block_size: BLOCK,
            lora_id: None,
            lora_name: None,
            cache_salt: None,
            is_bigram: false,
            medium: None,
        };
        provider
            .tree
            .store_event(&KvWorkerId::new(url.to_owned(), 0), &event);
    }

    fn lookup(provider: &RadixTreePrefixProvider, tokens: &[u32]) -> PrefixLookupResult {
        provider
            .match_request_tokens(tokens, &CacheNamespace::default())
            .expect("lookup")
    }

    fn depths(signal: &PrefixLookupResult) -> Vec<(String, u32)> {
        match &signal.outcome {
            PrefixOutcome::Matched { matches, .. } => {
                let mut v: Vec<_> = matches
                    .iter()
                    .map(|m| (m.address.clone(), m.matched_prefix_blocks))
                    .collect();
                v.sort();
                v
            }
            _ => Vec::new(),
        }
    }

    const IMAGE: u32 = 9_999;

    /// A provider with media affinity on and `IMAGE` as the image placeholder.
    fn media_provider() -> RadixTreePrefixProvider {
        media_provider_with_pending(60_000)
    }

    fn media_provider_with_pending(pending_ttl_ms: u64) -> RadixTreePrefixProvider {
        let provider = provider(pending_ttl_ms, 0);
        provider.tree.media().enable(Duration::from_secs(1_800));
        provider
            .tree
            .media_ranks()
            .enable(Duration::from_secs(1_800));
        provider.tree.set_media_tokens(vec![IMAGE]);
        provider
    }

    /// `pre` text blocks, then a block opening with a tagged image (the marker
    /// and the tokens after it stand in for its identity tag), then `post` blocks.
    fn media_prompt(pre: usize, post: usize) -> Vec<u32> {
        let mut prompt = tokens(pre);
        prompt.extend([IMAGE, MEDIA_TAG_MARK]);
        prompt.extend((0..(BLOCK as usize - 2) + post * BLOCK as usize).map(|i| 50_000 + i as u32));
        prompt
    }

    /// An image placeholder a caller's `input_ids` or a `/generate` prompt
    /// spells out carries no identity tag, so two such prompts with the same
    /// text and different images would share one placement: media affinity
    /// leaves them to route by their text.
    #[test]
    fn an_untagged_placeholder_is_not_a_media_prompt() {
        let provider = media_provider_with_pending(0);
        let ns = CacheNamespace::default();
        assert!(provider.tree.is_media_prompt(&media_prompt(4, 2)));
        let mut first = tokens(4);
        first.extend([IMAGE, 50_000, 50_001, 50_002, 50_003]);
        assert!(!provider.tree.is_media_prompt(&first));
        let signal = lookup(&provider, &first);
        assert!(signal.media_hashes.is_none());
        provider.record_route(&signal, "http://b");
        provider.record_rank(&first, &ns, "http://b", 1);
        let mut other = tokens(4);
        other.extend([IMAGE, 60_000, 60_001, 60_002, 60_003]);
        let signal = lookup(&provider, &other);
        assert!(matches!(signal.outcome, PrefixOutcome::Empty));
        assert!(provider
            .rank_depths(&other, &ns, "http://b")
            .depths
            .is_empty());
        assert!(provider
            .rank_depths(&first, &ns, "http://b")
            .depths
            .is_empty());
        assert_eq!(provider.tree.media().hits(), 0);
        assert_eq!(provider.tree.media_tiebreaks(), 0);
    }

    #[test]
    fn media_stake_routes_past_the_first_image() {
        let provider = media_provider();
        let turn2 = media_prompt(4, 6);
        let turn1 = &turn2[..(4 + 1 + 3) * BLOCK as usize];
        // Both workers hold the shared text before the image; KV events can confirm no more.
        confirm(&provider, "http://a", turn1, 4);
        confirm(&provider, "http://b", turn1, 4);
        let first = lookup(&provider, turn1);
        assert!(first.media_hashes.is_some());
        assert!(first.media_winners.is_empty(), "nothing remembered yet");
        provider.record_route(&first, "http://b");
        let signal = lookup(&provider, &turn2);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 4), ("http://b".into(), 8)]
        );
        assert_eq!(signal.media_winners, ["http://b"]);
        assert_eq!(provider.tree.media().hits(), 1);
        assert_eq!(
            provider.tree.pending().hits(),
            1,
            "the 60 s stake also saw b"
        );
    }

    #[test]
    fn deeper_confirmed_match_beats_a_media_stake() {
        let provider = media_provider();
        let prompt = media_prompt(4, 6);
        let turn1 = &prompt[..(4 + 1 + 1) * BLOCK as usize];
        let first = lookup(&provider, turn1);
        provider.record_route(&first, "http://b");
        confirm(&provider, "http://a", &prompt, 9);
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal)[0], ("http://a".into(), 9));
        assert_eq!(provider.tree.media().hits(), 0);
    }

    #[test]
    fn media_affinity_off_or_text_prompt_is_unchanged() {
        // Off: no media hashes, no tie-break.
        let off = provider(60_000, 0);
        let prompt = media_prompt(4, 2);
        confirm(&off, "http://a", &prompt, 4);
        confirm(&off, "http://b", &prompt, 4);
        let signal = lookup(&off, &prompt);
        assert!(signal.media_hashes.is_none());
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 4), ("http://b".into(), 4)]
        );
        // On, but a text prompt: also untouched.
        let on = media_provider();
        let text = tokens(8);
        confirm(&on, "http://a", &text, 4);
        confirm(&on, "http://b", &text, 4);
        let signal = lookup(&on, &text);
        assert!(signal.media_hashes.is_none());
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 4), ("http://b".into(), 4)]
        );
        assert_eq!(on.tree.media_tiebreaks(), 0);
    }

    #[test]
    fn routers_without_a_stake_break_the_tie_the_same_way() {
        // Two routers (separate trees) with the same confirmed state and no stakes.
        let prompt = media_prompt(4, 3);
        let winners: Vec<_> = (0..2)
            .map(|_| {
                let provider = media_provider();
                for url in ["http://a", "http://b", "http://c"] {
                    confirm(&provider, url, &prompt, 4);
                }
                let signal = lookup(&provider, &prompt);
                assert_eq!(provider.tree.media_tiebreaks(), 1);
                let boosted: Vec<_> = depths(&signal)
                    .into_iter()
                    .filter(|(_, d)| *d == 5)
                    .collect();
                assert_eq!(boosted.len(), 1, "exactly one worker is preferred");
                boosted[0].0.clone()
            })
            .collect();
        assert_eq!(winners[0], winners[1]);
        // A later turn of the same conversation keeps the key, so the same winner.
        let provider = media_provider();
        let later = media_prompt(4, 7);
        for url in ["http://a", "http://b", "http://c"] {
            confirm(&provider, url, &later, 4);
        }
        let signal = lookup(&provider, &later);
        assert!(depths(&signal).contains(&(winners[0].clone(), 5)));
    }

    #[test]
    fn media_rank_follows_the_recorded_rank() {
        let provider = media_provider();
        let prompt = media_prompt(4, 4);
        for rank in 0..2 {
            let event = BlockStored {
                block_hashes: provider
                    .block_hashes(&prompt, &CacheNamespace::default())
                    .unwrap()[..4]
                    .to_vec(),
                parent_block_hash: None,
                token_ids: prompt[..4 * BLOCK as usize].to_vec(),
                block_size: BLOCK,
                lora_id: None,
                lora_name: None,
                cache_salt: None,
                is_bigram: false,
                medium: None,
            };
            provider
                .tree
                .store_event(&KvWorkerId::new("http://a".to_owned(), rank), &event);
        }
        let ns = CacheNamespace::default();
        provider.record_rank(&prompt[..(4 + 2) * BLOCK as usize], &ns, "http://a", 1);
        let mut by_rank = provider.rank_depths(&prompt, &ns, "http://a").depths;
        by_rank.sort();
        assert_eq!(by_rank, [(0, 4), (1, 6)]);
        // Text prompts never record rank placements (a prompt sharing nothing with
        // the media chat, so only its own record could match).
        let text: Vec<u32> = (70_000..70_000 + 6 * BLOCK).collect();
        provider.record_rank(&text, &ns, "http://a", 0);
        assert!(provider
            .tree
            .media_ranks()
            .depths(&provider.block_hashes(&text, &ns).unwrap())
            .is_empty());
    }

    /// A first turn that ends soon after its image leaves the image's block
    /// partial; the next turn fills it, which moves that block's hash. The
    /// placement and the conversation key must survive that.
    #[test]
    fn placement_survives_an_image_in_a_partial_block() {
        const WIDE: u32 = 8;
        let tree = Arc::new(HashTree::new());
        let oracle = BlockSizeOracle::new();
        oracle.try_set(WIDE).expect("block size");
        tree.media().enable(Duration::from_secs(1_800));
        tree.media_ranks().enable(Duration::from_secs(1_800));
        tree.set_media_tokens(vec![IMAGE]);
        let provider = RadixTreePrefixProvider::new(tree, oracle);
        let ns = CacheNamespace::default();
        // Four text blocks, then the image and its tag open a fifth block that
        // the turn does not fill.
        let mut turn1: Vec<u32> = (1..=32).collect();
        turn1.extend([IMAGE, MEDIA_TAG_MARK, 50_000, 50_001]);
        let mut turn2 = turn1.clone();
        turn2.extend(50_002..50_022);
        let hashes1 = provider.block_hashes(&turn1, &ns).unwrap();
        let hashes2 = provider.block_hashes(&turn2, &ns).unwrap();
        assert_ne!(hashes1[4], hashes2[4], "the image block's own hash moves");
        let media1 = provider.media_chain(&turn1, &ns, &hashes1).unwrap();
        let media2 = provider.media_chain(&turn2, &ns, &hashes2).unwrap();
        assert_eq!(media1.key, media2.key, "the conversation key does not");
        assert_eq!(
            media1.chain,
            [media1.key],
            "turn 1 ends in the image's block"
        );
        assert_eq!(media2.chain, [media2.key, hashes2[5], hashes2[6]]);
        for url in ["http://a", "http://b"] {
            let event = BlockStored {
                block_hashes: hashes1[..4].to_vec(),
                parent_block_hash: None,
                token_ids: turn1[..32].to_vec(),
                block_size: WIDE,
                lora_id: None,
                lora_name: None,
                cache_salt: None,
                is_bigram: false,
                medium: None,
            };
            provider
                .tree
                .store_event(&KvWorkerId::new(url.to_owned(), 0), &event);
        }
        let first = lookup(&provider, &turn1);
        assert_eq!(provider.tree.media_tiebreaks(), 1, "no placement yet");
        // The preference is one block over the confirmed four, which the partial
        // image block makes the fifth query block: the policy's cap at
        // `query_blocks` leaves it intact.
        let mut first_depths: Vec<_> = depths(&first).into_iter().map(|(_, d)| d).collect();
        first_depths.sort();
        assert_eq!(first_depths, [4, 5]);
        assert_eq!(first.query_blocks, 5);
        provider.record_route(&first, "http://b");
        provider.record_rank(&turn1, &ns, "http://b", 1);
        let signal = lookup(&provider, &turn2);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 4), ("http://b".into(), 5)]
        );
        assert_eq!(provider.tree.media().hits(), 1);
        assert_eq!(
            provider.tree.media_tiebreaks(),
            1,
            "remembered: no tie-break"
        );
        let mut by_rank = provider.rank_depths(&turn2, &ns, "http://b").depths;
        by_rank.sort();
        assert_eq!(by_rank, [(0, 4), (1, 5)]);
    }

    /// An image chat shorter than one KV block is hashed as a partial block,
    /// so its placement is remembered like any other.
    #[test]
    fn a_chat_shorter_than_a_block_keeps_its_worker() {
        const WIDE: u32 = 8;
        let tree = Arc::new(HashTree::new());
        let oracle = BlockSizeOracle::new();
        oracle.try_set(WIDE).expect("block size");
        tree.media().enable(Duration::from_secs(1_800));
        tree.media_ranks().enable(Duration::from_secs(1_800));
        tree.set_media_tokens(vec![IMAGE]);
        let provider = RadixTreePrefixProvider::new(tree, oracle);
        let ns = CacheNamespace::default();
        let turn1 = vec![1, 2, IMAGE, MEDIA_TAG_MARK, 50_000, 50_001];
        let first = lookup(&provider, &turn1);
        assert!(matches!(first.outcome, PrefixOutcome::Empty));
        assert_eq!(first.query_blocks, 1);
        provider.record_route(&first, "http://b");
        provider.record_rank(&turn1, &ns, "http://b", 1);
        let mut turn2 = turn1.clone();
        turn2.extend([50_002, 50_003]);
        let signal = lookup(&provider, &turn2);
        assert_eq!(depths(&signal), [("http://b".into(), 1)]);
        assert_eq!(signal.media_winners, ["http://b"]);
        assert_eq!(provider.tree.media().hits(), 1);
        assert_eq!(
            provider.rank_depths(&turn2, &ns, "http://b").depths,
            [(1, 1)]
        );
    }

    /// Ranks tied on the confirmed prefix with none remembered: the rendezvous
    /// rank is a preference every replica shares, not a deeper depth, so rank
    /// load still decides first.
    #[test]
    fn rank_preference_is_a_tie_break_not_a_depth() {
        let prompt = media_prompt(4, 2);
        let ns = CacheNamespace::default();
        let preferred: Vec<_> = (0..2)
            .map(|_| {
                let provider = media_provider();
                for rank in [0, 1] {
                    let event = BlockStored {
                        block_hashes: provider.block_hashes(&prompt, &ns).unwrap()[..4].to_vec(),
                        parent_block_hash: None,
                        token_ids: prompt[..4 * BLOCK as usize].to_vec(),
                        block_size: BLOCK,
                        lora_id: None,
                        lora_name: None,
                        cache_salt: None,
                        is_bigram: false,
                        medium: None,
                    };
                    provider
                        .tree
                        .store_event(&KvWorkerId::new("http://a".to_owned(), rank), &event);
                }
                let placement = provider.rank_depths(&prompt, &ns, "http://a");
                let mut depths = placement.depths.clone();
                depths.sort();
                assert_eq!(depths, [(0, 4), (1, 4)], "no rank is boosted");
                assert_eq!(
                    provider.tree.media_tiebreaks(),
                    0,
                    "a rank preference is not a tie-break: load may not follow it"
                );
                let preferred = placement.preferred.expect("a tied rank is preferred");
                // Once a rank is remembered, it is the depth that decides.
                provider.record_rank(&prompt, &ns, "http://a", 1);
                let placement = provider.rank_depths(&prompt, &ns, "http://a");
                let mut depths = placement.depths.clone();
                depths.sort();
                assert_eq!(depths, [(0, 4), (1, 7)]);
                assert_eq!(placement.preferred, None);
                preferred
            })
            .collect();
        assert_eq!(
            preferred[0], preferred[1],
            "every replica prefers the same rank"
        );
    }

    /// A burst of new image chats on one cold system prompt: the first chat's
    /// placement must not attract the others, which share only that text.
    #[test]
    fn a_chat_sharing_only_the_text_before_its_image_inherits_no_placement() {
        let provider = media_provider_with_pending(0);
        let ns = CacheNamespace::default();
        let first = media_prompt(4, 2);
        let signal = lookup(&provider, &first);
        assert!(matches!(signal.outcome, PrefixOutcome::Empty));
        provider.record_route(&signal, "http://b");
        provider.record_rank(&first, &ns, "http://b", 1);
        let mut other = first[..4 * BLOCK as usize].to_vec();
        other.extend([
            IMAGE,
            MEDIA_TAG_MARK,
            60_000,
            60_001,
            60_002,
            60_003,
            60_004,
        ]);
        let signal = lookup(&provider, &other);
        assert!(
            matches!(signal.outcome, PrefixOutcome::Empty),
            "nothing remembered for it"
        );
        assert_eq!(provider.tree.media().hits(), 0);
        assert_eq!(provider.tree.media_tiebreaks(), 0);
        assert!(provider
            .rank_depths(&other, &ns, "http://b")
            .depths
            .is_empty());
        // The first chat's own next turn still finds its placement.
        let next = media_prompt(4, 5);
        let signal = lookup(&provider, &next);
        assert_eq!(depths(&signal), [("http://b".into(), 7)]);
        assert_eq!(provider.tree.media().hits(), 1);
        assert_eq!(
            provider.rank_depths(&next, &ns, "http://b").depths,
            [(1, 7)]
        );
    }

    #[test]
    fn equal_depth_rank_placement_never_outranks_a_confirmed_rank() {
        let provider = media_provider();
        let ns = CacheNamespace::default();
        let prompt = media_prompt(4, 2);
        // Rank 0 of a holds the four text blocks.
        let event = BlockStored {
            block_hashes: provider.block_hashes(&prompt, &ns).unwrap()[..4].to_vec(),
            parent_block_hash: None,
            token_ids: prompt[..4 * BLOCK as usize].to_vec(),
            block_size: BLOCK,
            lora_id: None,
            lora_name: None,
            cache_salt: None,
            is_bigram: false,
            medium: None,
        };
        provider
            .tree
            .store_event(&KvWorkerId::new("http://a".to_owned(), 0), &event);
        // Rank 1 remembers another chat that shares only that text (another image).
        let mut other = prompt[..4 * BLOCK as usize].to_vec();
        other.extend([IMAGE, MEDIA_TAG_MARK, 60_000, 60_001, 60_002]);
        provider.record_rank(&other, &ns, "http://a", 1);
        assert_eq!(
            provider.rank_depths(&prompt, &ns, "http://a").depths,
            [(0, 4)]
        );
        assert_eq!(
            provider.tree.media_tiebreaks(),
            0,
            "one confirmed rank is no tie"
        );
    }

    #[test]
    fn confirmed_match_wins_over_a_shallower_or_equal_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 6);
        // A stake on b covering 4 blocks: shallower than a's confirmed 6.
        let probe = lookup(&provider, &prompt[..4 * BLOCK as usize]);
        provider.record_route(&probe, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 6)]);
        assert!(signal.pending_winners.is_empty());
        assert_eq!(provider.tree.pending().hits(), 0);
        // Equal depth on another worker: still the confirmed tree decides.
        let probe = lookup(&provider, &prompt[..6 * BLOCK as usize]);
        provider.record_route(&probe, "http://c");
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 6)]);
        assert!(signal.pending_winners.is_empty());
    }

    #[test]
    fn without_the_switch_pending_merges_by_maximum_as_before() {
        let provider = provider(60_000, 0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 6);
        // An equal-depth stake on another worker stays a candidate, as it did
        // before confirmed-first mode existed, and every request stakes.
        let probe = lookup(&provider, &prompt[..6 * BLOCK as usize]);
        assert!(probe.block_hashes.is_some());
        provider.record_route(&probe, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 6), ("http://b".into(), 6)]
        );
        assert!(signal.block_hashes.is_some(), "every request stakes");
        assert_eq!(
            provider.tree.pending().hits(),
            0,
            "not deeper than confirmed"
        );
    }

    #[test]
    fn deeper_stake_is_consulted_and_marked_as_pending_winner() {
        let provider = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 3);
        let signal = lookup(&provider, &prompt);
        provider.record_route(&signal, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 3), ("http://b".into(), 10)]
        );
        assert_eq!(signal.pending_winners, vec!["http://b".to_string()]);
        assert_eq!(provider.tree.pending().hits(), 1);
    }

    #[test]
    fn only_strictly_deeper_stakes_join_the_candidates() {
        let first = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&first, "http://a", &prompt, 2);
        // c stakes 3 blocks (deeper than a's 2), d stakes 2 (equal) and e 1.
        for (url, blocks) in [("http://c", 3), ("http://d", 2), ("http://e", 1)] {
            let probe = lookup(&first, &prompt[..blocks * BLOCK as usize]);
            first.record_route(&probe, url);
        }
        let signal = lookup(&first, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 2), ("http://c".into(), 3)],
            "the stakes no deeper than a's match are not candidates"
        );
        assert_eq!(signal.pending_winners, vec!["http://c".to_string()]);
        // Without the switch, every stake merges by maximum as before.
        let plain = provider(60_000, 0);
        confirm(&plain, "http://a", &prompt, 2);
        for (url, blocks) in [("http://c", 3), ("http://d", 2), ("http://e", 1)] {
            let probe = lookup(&plain, &prompt[..blocks * BLOCK as usize]);
            plain.record_route(&probe, url);
        }
        let signal = lookup(&plain, &prompt);
        assert_eq!(
            depths(&signal),
            [
                ("http://a".into(), 2),
                ("http://c".into(), 3),
                ("http://d".into(), 2),
                ("http://e".into(), 1)
            ]
        );
    }

    #[test]
    fn follower_cap_falls_back_to_the_confirmed_tree() {
        let provider = provider(60_000, 2).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 3);
        let first = lookup(&provider, &prompt);
        provider.record_route(&first, "http://b");
        // Two followers may ride the stake.
        for _ in 0..2 {
            let signal = lookup(&provider, &prompt);
            assert_eq!(signal.pending_winners, vec!["http://b".to_string()]);
            provider.record_route(&signal, "http://b");
        }
        assert_eq!(provider.tree.pending().followed(), 2);
        // The third sees only the confirmed tree: a at 3 blocks, b gone.
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 3)]);
        assert!(signal.pending_winners.is_empty());
        assert_eq!(provider.tree.pending().capped(), 1);
        // Once b's events arrive, b is a confirmed match again.
        confirm(&provider, "http://b", &prompt, 10);
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 3), ("http://b".into(), 10)]
        );
        assert!(signal.pending_winners.is_empty());
    }

    #[test]
    fn warm_prompt_skips_the_pending_tier_and_its_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(0.5);
        let prompt = tokens(10);
        // A cold prompt (nothing confirmed) consults the tier and is staked.
        let cold = lookup(&provider, &prompt);
        assert!(cold.block_hashes.is_some(), "a cold prompt is staked");
        provider.record_route(&cold, "http://b");
        // Once a confirmed match covers at least half the prompt, the deeper
        // stake on b is not even consulted, and the prompt writes no stake.
        confirm(&provider, "http://a", &prompt, 6);
        let warm = lookup(&provider, &prompt);
        assert_eq!(depths(&warm), [("http://a".into(), 6)]);
        assert!(warm.pending_winners.is_empty());
        assert!(warm.block_hashes.is_none(), "a warm prompt pays no stake");
        assert_eq!(
            provider.tree.pending().hits(),
            0,
            "the tier was never walked"
        );
    }

    #[test]
    fn cold_prompt_below_the_gate_still_uses_the_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(0.5);
        let prompt = tokens(10);
        // Only the shared head is confirmed: 2 of 10 blocks, below the gate.
        confirm(&provider, "http://a", &prompt, 2);
        let first = lookup(&provider, &prompt);
        assert!(first.block_hashes.is_some());
        provider.record_route(&first, "http://b");
        let next = lookup(&provider, &prompt);
        assert_eq!(next.pending_winners, vec!["http://b".to_string()]);
        assert_eq!(
            depths(&next),
            [("http://a".into(), 2), ("http://b".into(), 10)]
        );
        assert_eq!(provider.tree.pending().hits(), 1);
    }

    #[test]
    fn routing_to_a_confirmed_worker_is_not_a_follow() {
        let provider = provider(60_000, 1);
        let prompt = tokens(4);
        confirm(&provider, "http://a", &prompt, 4);
        for _ in 0..3 {
            let signal = lookup(&provider, &prompt);
            assert!(signal.pending_winners.is_empty());
            provider.record_route(&signal, "http://a");
        }
        assert_eq!(provider.tree.pending().followed(), 0);
        assert_eq!(provider.tree.pending().capped(), 0);
    }
}
