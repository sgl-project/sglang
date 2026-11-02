// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Translate worker-local LoRA block IDs into adapter-aware prompt lookup keys.
//! Base-model events bypass this module. An external ID owns a set of tiers;
//! several IDs may own the same lookup key without evicting each other.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraBlock {
    pub external: i64,
    pub lookup: i64,
    pub digest: [u8; 32],
    pub parent: Option<i64>,
    pub tiers: u8,
    pub name: String,
    pub salt: Option<String>,
}

#[derive(Clone, Debug)]
pub struct LoraStore {
    pub hashes: Vec<i64>,
    pub parent: Option<i64>,
    /// Already-flattened token atoms: bigrams contribute two u32s per atom.
    pub tokens: Vec<u32>,
    pub block_size: usize,
    pub bigram: bool,
    pub name: String,
    pub salt: Option<String>,
    pub tiers: u8,
    pub component_mask: Option<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct LookupStore {
    pub parent: Option<i64>,
    pub hashes: Vec<i64>,
    pub tiers: u8,
    pub component_mask: Option<u32>,
    pub block_size: usize,
}

#[derive(Debug, Default)]
pub struct LoraIndex {
    blocks: HashMap<i64, LoraBlock>,
    owners: HashMap<i64, [usize; 4]>,
    children: HashMap<i64, usize>,
    pending: Vec<LoraStore>,
}

pub fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for part in parts {
        h.update(part);
    }
    h.finalize().into()
}

pub fn lookup_hash(digest_value: &[u8; 32], name: &str) -> i64 {
    let seed = digest(&[b"sglang-kv-event-lora-v1\0", name.as_bytes()]);
    i64::from_be_bytes(
        digest(&[&seed, &digest_value[..8]])[..8]
            .try_into()
            .unwrap(),
    )
}

impl LoraIndex {
    pub fn has_blocks(&self) -> bool {
        !self.blocks.is_empty()
    }

    pub fn snapshot(&self) -> Vec<LoraBlock> {
        self.blocks.values().cloned().collect()
    }

    /// Snapshot callers validate worker identities and corresponding tree carriers.
    pub fn restore(&mut self, blocks: &[LoraBlock]) {
        for block in blocks {
            if let Some(old) = self.blocks.get(&block.external) {
                if old.lookup != block.lookup || old.digest != block.digest {
                    continue;
                }
            } else {
                if let Some(parent) = block.parent {
                    *self.children.entry(parent).or_default() += 1;
                }
                let mut entry = block.clone();
                entry.tiers = 0;
                self.blocks.insert(block.external, entry);
            }
            self.add_tiers(block.external, block.tiers & 15);
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Repeated stores are idempotent, including replay after a snapshot.
    /// Unknown parents wait for their store; malformed chains never get credit.
    pub fn store(&mut self, event: LoraStore) -> Vec<LookupStore> {
        let mut out = Vec::new();
        if event.tiers & 15 == 0 || event.hashes.is_empty() || !Self::valid(&event) {
            return out;
        }
        if let Some(stored) = self.try_store(&event) {
            out.push(stored);
        } else if self.pending.len() < 1024
            && event.tokens.len() + self.pending.iter().map(|e| e.tokens.len()).sum::<usize>()
                <= 1_048_576
        {
            self.pending.push(event);
        }
        loop {
            let pending = std::mem::take(&mut self.pending);
            let mut progressed = false;
            for event in pending {
                if let Some(stored) = self.try_store(&event) {
                    out.push(stored);
                    progressed = true;
                } else {
                    self.pending.push(event);
                }
            }
            if !progressed {
                break;
            }
        }
        out
    }

    fn valid(event: &LoraStore) -> bool {
        let Some(width) = event
            .block_size
            .checked_mul(if event.bigram { 2 } else { 1 })
        else {
            return false;
        };
        width > 0
            && !event.name.is_empty()
            && !event.tokens.is_empty()
            && event.tokens.len().div_ceil(width) == event.hashes.len()
            && (!event.bigram || event.tokens.len().is_multiple_of(2))
    }

    fn try_store(&mut self, event: &LoraStore) -> Option<LookupStore> {
        let parent = match event.parent {
            Some(id) => {
                let block = self.blocks.get(&id)?;
                if block.name != event.name || block.salt != event.salt {
                    return None;
                }
                Some(block)
            }
            None => None,
        };
        let lookup_parent = parent.map(|b| b.lookup);
        let mut prior = parent.map(|b| b.digest).or_else(|| {
            event
                .salt
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(|salt| digest(&[b"sglang-cache-salt-v1\0", salt.as_bytes()]))
        });
        let width = event.block_size * if event.bigram { 2 } else { 1 };
        let mut external_parent = event.parent;
        let mut blocks = Vec::with_capacity(event.hashes.len());
        let mut seen = HashSet::with_capacity(event.hashes.len());
        // Validate the whole chain before changing any ownership.
        for (&external, tokens) in event.hashes.iter().zip(event.tokens.chunks(width)) {
            let mut h = Sha256::new();
            if let Some(p) = prior {
                h.update(p);
            }
            for token in tokens {
                h.update(token.to_le_bytes());
            }
            let d: [u8; 32] = h.finalize().into();
            let lookup = lookup_hash(&d, &event.name);
            if external_parent == Some(external)
                || !seen.insert(external)
                || self.blocks.get(&external).is_some_and(|old| {
                    old.lookup != lookup || old.digest != d || old.parent != external_parent
                })
            {
                return None;
            }
            blocks.push(LoraBlock {
                external,
                lookup,
                digest: d,
                parent: external_parent,
                tiers: 0,
                name: event.name.clone(),
                salt: event.salt.clone(),
            });
            prior = Some(d);
            external_parent = Some(external);
        }
        let hashes = blocks.iter().map(|b| b.lookup).collect();
        for block in blocks {
            if !self.blocks.contains_key(&block.external) {
                if let Some(parent) = block.parent {
                    *self.children.entry(parent).or_default() += 1;
                }
                self.blocks.insert(block.external, block.clone());
            }
            self.add_tiers(block.external, event.tiers & 15);
        }
        Some(LookupStore {
            parent: lookup_parent,
            hashes,
            tiers: event.tiers & 15,
            component_mask: event.component_mask,
            block_size: event.block_size,
        })
    }

    fn add_tiers(&mut self, external: i64, tiers: u8) {
        let block = self.blocks.get_mut(&external).unwrap();
        let added = tiers & !block.tiers;
        if added == 0 {
            return;
        }
        let counts = self.owners.entry(block.lookup).or_default();
        for (i, count) in counts.iter_mut().enumerate() {
            if added & (1 << i) != 0 {
                *count += 1;
            }
        }
        block.tiers |= tiers;
    }

    /// `(lookup key, tiers that lost their last owner)`; unknown IDs pass
    /// through so base-model removals retain their existing behavior.
    pub fn remove(&mut self, hashes: &[i64], tiers: u8) -> Vec<(i64, u8)> {
        for event in &mut self.pending {
            if event.hashes.iter().any(|h| hashes.contains(h)) {
                event.tiers &= !tiers;
            }
        }
        self.pending.retain(|e| e.tiers != 0);
        let mut out = Vec::new();
        for &external in hashes {
            let Some(block) = self.blocks.get_mut(&external) else {
                out.push((external, tiers));
                continue;
            };
            let removed = block.tiers & tiers;
            block.tiers &= !tiers;
            if removed == 0 {
                self.prune(external);
                continue;
            }
            let counts = self.owners.get_mut(&block.lookup).unwrap();
            let mut lost = 0;
            for (i, count) in counts.iter_mut().enumerate() {
                if removed & (1 << i) != 0 {
                    *count -= 1;
                    if *count == 0 {
                        lost |= 1 << i;
                    }
                }
            }
            if lost != 0 {
                out.push((block.lookup, lost));
            }
            if counts.iter().all(|&n| n == 0) {
                self.owners.remove(&block.lookup);
            }
            self.prune(external);
        }
        out
    }

    fn prune(&mut self, mut id: i64) {
        while self.blocks.get(&id).is_some_and(|b| b.tiers == 0)
            && self.children.get(&id).copied().unwrap_or(0) == 0
        {
            let block = self.blocks.remove(&id).unwrap();
            self.children.remove(&id);
            let Some(parent) = block.parent else { break };
            if let Some(count) = self.children.get_mut(&parent) {
                *count -= 1;
            }
            id = parent;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(ids: &[i64], tokens: &[u32]) -> LoraStore {
        LoraStore {
            hashes: ids.to_vec(),
            parent: None,
            tokens: tokens.to_vec(),
            block_size: 2,
            bigram: false,
            name: "adapter-a".into(),
            salt: None,
            tiers: 1,
            component_mask: None,
        }
    }

    #[test]
    fn opaque_load_ids_share_lookup_but_keep_independent_tier_ownership() {
        let mut map = LoraIndex::default();
        let first = store(&[100, 101], &[1, 2, 3, 4]);
        let keys = map.store(first.clone())[0].hashes.clone();
        // Independent Python fixture values, rather than recomputing expectations.
        assert_eq!(keys, [1132550695741458728, 3694048958605100876]);
        assert_eq!(map.store(store(&[200, 201], &[1, 2, 3, 4]))[0].hashes, keys);
        // Duplicate stores/replay must not create an additional owner.
        map.store(first.clone());
        let mut host = first;
        host.tiers = 2;
        map.store(host);
        assert!(map.remove(&[100, 101], 1).is_empty());
        assert_eq!(map.remove(&[200, 201], 1), vec![(keys[0], 1), (keys[1], 1)]);
        assert_eq!(map.remove(&[100, 101], 2), vec![(keys[0], 2), (keys[1], 2)]);
        assert!(map.snapshot().is_empty(), "dead aliases must be reclaimed");
    }

    #[test]
    fn continuation_orphan_and_snapshot_replay_preserve_parent_identity() {
        let mut map = LoraIndex::default();
        let mut child = store(&[101], &[3, 4]);
        child.parent = Some(100);
        assert!(map.store(child.clone()).is_empty());
        let stored = map.store(store(&[100], &[1, 2]));
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[1].parent, Some(stored[0].hashes[0]));
        assert_eq!(stored[1].hashes, [3694048958605100876]);
        let mut restored = LoraIndex::default();
        restored.restore(&map.snapshot());
        restored.store(child);
        assert_eq!(restored.remove(&[101], 15), vec![(3694048958605100876, 1)]);
        assert_eq!(restored.remove(&[100], 15), vec![(1132550695741458728, 1)]);
        assert!(restored.snapshot().is_empty());
        // Non-LoRA removals keep their original IDs and tier masks.
        assert_eq!(restored.remove(&[42], 15), vec![(42, 15)]);
    }

    #[test]
    fn removed_parent_digest_survives_while_a_child_owns_it() {
        let mut map = LoraIndex::default();
        map.store(store(&[100, 101], &[1, 2, 3, 4]));
        map.remove(&[100], 1);
        assert!(map.remove(&[100], 1).is_empty());
        let mut branch = store(&[102], &[5, 6]);
        branch.parent = Some(100);
        assert_eq!(map.store(branch)[0].parent, Some(1132550695741458728));
        map.clear();
        assert!(map.snapshot().is_empty());
    }

    #[test]
    fn salted_and_bigram_stores_use_the_request_token_atoms() {
        let mut map = LoraIndex::default();
        let mut salted = store(&[100, 101], &[1, 2, 3, 4]);
        salted.salt = Some("tenant-a".into());
        assert_eq!(
            map.store(salted)[0].hashes,
            [5453732947292851540, -615216578871236088]
        );
        let mut bigram = store(&[200, 201], &[1, 2, 2, 3, 3, 4]);
        bigram.bigram = true;
        let keys = map.store(bigram)[0].hashes.clone();
        let mut other_adapter = store(&[300, 301], &[1, 2, 2, 3, 3, 4]);
        other_adapter.bigram = true;
        other_adapter.name = "adapter-b".into();
        assert_ne!(map.store(other_adapter)[0].hashes, keys);
        let mut invalid = store(&[400, 401], &[1, 2]);
        assert!(map.store(invalid.clone()).is_empty());
        invalid.block_size = 0;
        assert!(map.store(invalid).is_empty());
    }
}
