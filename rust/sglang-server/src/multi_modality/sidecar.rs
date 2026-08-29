//! Rid-keyed parking of finished results between an MM worker and the
//! scheduler drain.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::shm::{ShmSegment, shm_name};

/// One parked result: the buffers the drain-time Python adapter needs (the
/// expanded `input_ids` travel separately, via `TmEvent::MmEncoded`). The qwen
/// drain shape (`sglang_mm::qwen_vl::pack_drain`); generalizes to a
/// named-tensor handoff once a family needs a different one.
///
/// Constructed from outside the module only by tests; the worker parks every
/// real entry itself.
pub struct MmSidecarEntry {
    pub features: FeatureStore,
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

/// Where a result's feature buffers live between worker and drain.
pub enum FeatureStore {
    /// In-process; the drain wraps them zero-copy. Single-rank serving, or the
    /// shm fallback. Under TP the whole buffer would ride `broadcast_pyobj`.
    Inline(Vec<f32>),
    /// One POSIX segment per item, written by the worker; only the names cross
    /// ranks. See [`ShmSegment`].
    Shm(Vec<ShmSegment>),
}

/// Results parked between a worker's `MmEncoded` and the scheduler drain, keyed
/// by rid. Owns the lifecycle so entries never leak: [`park`](Self::park)
/// strictly before `MmEncoded`, [`take`](Self::take) at the drain,
/// [`purge`](Self::purge) for requests that die while parked.
#[derive(Clone, Default)]
pub struct Sidecar(Arc<Mutex<HashMap<String, MmSidecarEntry>>>);

impl Sidecar {
    pub fn park(&self, rid: String, entry: MmSidecarEntry) {
        self.0.lock().unwrap().insert(rid, entry);
    }
    pub fn take(&self, rid: &str) -> Option<MmSidecarEntry> {
        self.0.lock().unwrap().remove(rid)
    }
    pub fn purge(&self, rid: &str) {
        self.0.lock().unwrap().remove(rid);
    }
}

/// Split the flat feature buffer per item (`t*h*w` rows per grid) and park each
/// slice in its own segment. Any shm failure (`/dev/shm` full, odd shape) falls
/// back to inline, as Python's `_wrap_shm_or_inline` does: degrade to the slow
/// path, never fail the request.
pub(super) fn park_features_in_shm(features: &[f32], grids: &[[u32; 3]]) -> FeatureStore {
    let total_rows: usize = grids
        .iter()
        .map(|g| g[0] as usize * g[1] as usize * g[2] as usize)
        .sum();
    if total_rows == 0 || !features.len().is_multiple_of(total_rows) {
        return FeatureStore::Inline(features.to_vec());
    }
    let dim = features.len() / total_rows;
    let mut segments = Vec::with_capacity(grids.len());
    let mut row = 0usize;
    for (item, grid) in grids.iter().enumerate() {
        let rows = grid[0] as usize * grid[1] as usize * grid[2] as usize;
        let slice = &features[row * dim..(row + rows) * dim];
        row += rows;
        match ShmSegment::create(shm_name(item), bytemuck::cast_slice(slice)) {
            Ok(segment) => segments.push(segment),
            Err(error) => {
                tracing::warn!(%error, "mm: shm feature transport failed; falling back to inline");
                return FeatureStore::Inline(features.to_vec());
            }
        }
    }
    FeatureStore::Shm(segments)
}

#[cfg(test)]
mod tests {
    use super::super::shm::shm_path;
    use super::*;

    /// Per-item slicing follows the grid row counts, so Python's
    /// `(rows, feature_dim)` reshape of a segment sees only its own item.
    #[test]
    fn park_splits_features_by_grid() {
        // Two items: grids (1,2,2)=4 rows and (1,1,2)=2 rows, dim=3.
        let features: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let grids = [[1, 2, 2], [1, 1, 2]];
        let FeatureStore::Shm(segments) = park_features_in_shm(&features, &grids) else {
            panic!("expected shm store");
        };
        assert_eq!(segments.len(), 2);
        let read = |seg: &ShmSegment| -> Vec<u8> { std::fs::read(shm_path(&seg.name)).unwrap() };
        assert_eq!(
            read(&segments[0]),
            bytemuck::cast_slice::<f32, u8>(&features[..12])
        );
        assert_eq!(
            read(&segments[1]),
            bytemuck::cast_slice::<f32, u8>(&features[12..])
        );
    }

    /// A degenerate shape must degrade to inline, never a shm-side panic.
    #[test]
    fn shape_surprise_falls_back_inline() {
        let features = vec![0.0f32; 7]; // not divisible by 2 rows
        let grids = [[1, 1, 2]];
        assert!(matches!(
            park_features_in_shm(&features, &grids),
            FeatureStore::Inline(_)
        ));
    }
}
