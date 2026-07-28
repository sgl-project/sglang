use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::mooncake::{EngineOwner, MemoryBuffer, MemoryLocation, Region as MooncakeRegion};
use crate::pd::buffer::BufferError;
use crate::pd::protocol::{FixedBytes, RegionRecord};
use crate::pd::room::RegistrationEpoch;

pub const REGION_COUNT: usize = 58;
pub const KV_REGION_COUNT: usize = 56;
pub const KV_PAGE_SIZE_TOKENS: u32 = 64;
pub const KV_ROW_BYTES: u64 = 2_048;
pub const KV_PAGE_BYTES: u64 = 131_072;
pub const AUX_SLOT_BYTES: u64 = 64;
pub const COMPLETION_SLOT_BYTES: u64 = 192;
pub const SLOT_COUNT: u64 = 32;
const REGION_ALIGNMENT: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    Key { layer: u16 },
    Value { layer: u16 },
    Aux,
    Completion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionLocation {
    Device { device: u32 },
    PinnedHost { numa_node: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferDType {
    BFloat16,
    Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionLayout {
    pub shape: Vec<u64>,
    pub strides_bytes: Vec<u64>,
    pub page_size_tokens: u32,
    pub item_size_bytes: u32,
}

impl RegionLayout {
    pub fn kv(page_capacity: u64) -> Result<Self, BufferError> {
        if page_capacity == 0 {
            return invalid("layout.shape", "KV page capacity must be non-zero");
        }
        Ok(Self {
            shape: vec![page_capacity, 64, 8, 128],
            strides_bytes: vec![KV_PAGE_BYTES, KV_ROW_BYTES, 256, 2],
            page_size_tokens: KV_PAGE_SIZE_TOKENS,
            item_size_bytes: 2,
        })
    }

    pub fn aux() -> Self {
        Self {
            shape: vec![SLOT_COUNT, AUX_SLOT_BYTES],
            strides_bytes: vec![AUX_SLOT_BYTES, 1],
            page_size_tokens: 0,
            item_size_bytes: 1,
        }
    }

    pub fn completion() -> Self {
        Self {
            shape: vec![SLOT_COUNT, COMPLETION_SLOT_BYTES],
            strides_bytes: vec![COMPLETION_SLOT_BYTES, 1],
            page_size_tokens: 0,
            item_size_bytes: 1,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct BufferRegionSpec {
    pub region_id: u16,
    pub kind: RegionKind,
    pub base_address: u64,
    pub length_bytes: u64,
    pub location: RegionLocation,
    pub dtype: BufferDType,
    pub layout: RegionLayout,
    pub owner_generation: u64,
    pub layout_fingerprint: FixedBytes<32>,
}

impl fmt::Debug for BufferRegionSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BufferRegionSpec")
            .field("region_id", &self.region_id)
            .field("kind", &self.kind)
            .field("length_bytes", &self.length_bytes)
            .field("location", &self.location)
            .field("dtype", &self.dtype)
            .field("layout", &self.layout)
            .field("owner_generation", &self.owner_generation)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct BufferTable {
    regions: Vec<BufferRegionSpec>,
    owner_generation: u64,
    device: u32,
    layout_fingerprint: FixedBytes<32>,
    kv_page_capacity: u64,
}

impl BufferTable {
    pub fn new(
        regions: Vec<BufferRegionSpec>,
        owner_generation: u64,
        device: u32,
        layout_fingerprint: FixedBytes<32>,
    ) -> Result<Self, BufferError> {
        if regions.len() != REGION_COUNT {
            return invalid("regions", "the table must contain exactly 58 regions");
        }
        if owner_generation == 0 {
            return invalid("owner_generation", "must be non-zero");
        }
        if !matches!(device, 4 | 5) {
            return invalid("device", "must be the frozen prefill or decode GPU");
        }
        if layout_fingerprint.as_bytes().iter().all(|byte| *byte == 0) {
            return invalid("layout_fingerprint", "must not be all zero");
        }

        let mut ids = BTreeSet::new();
        let mut kv_page_capacity = None;
        for (expected_id, region) in (0_u16..58).zip(&regions) {
            if region.region_id != expected_id || !ids.insert(region.region_id) {
                return invalid(
                    "regions",
                    "regions must be complete and ordered by unique RegionId",
                );
            }
            validate_region(
                region,
                owner_generation,
                device,
                layout_fingerprint,
                &mut kv_page_capacity,
            )?;
        }

        Ok(Self {
            regions,
            owner_generation,
            device,
            layout_fingerprint,
            kv_page_capacity: kv_page_capacity.expect("58-region table contains KV regions"),
        })
    }

    pub fn regions(&self) -> &[BufferRegionSpec] {
        &self.regions
    }

    pub fn region(&self, region_id: u16) -> Result<&BufferRegionSpec, BufferError> {
        self.regions
            .get(usize::from(region_id))
            .filter(|region| region.region_id == region_id)
            .ok_or(BufferError::InvalidDescriptor {
                field: "region_id",
                detail: "is outside the frozen mapping",
            })
    }

    pub const fn owner_generation(&self) -> u64 {
        self.owner_generation
    }

    pub const fn device(&self) -> u32 {
        self.device
    }

    pub const fn layout_fingerprint(&self) -> FixedBytes<32> {
        self.layout_fingerprint
    }

    pub const fn kv_page_capacity(&self) -> u64 {
        self.kv_page_capacity
    }

    pub fn register<P>(self, port: &mut P) -> Result<RegisteredRegionTable<P::Handle>, BufferError>
    where
        P: RegistrationPort,
    {
        let mut handles = Vec::with_capacity(REGION_COUNT);
        for region in &self.regions {
            match port.register(region) {
                Ok(handle) => handles.push(Some(handle)),
                Err(_) => {
                    let mut rollback_failures = 0;
                    for handle in handles.drain(..).rev().flatten() {
                        if port.unregister(handle).is_err() {
                            rollback_failures += 1;
                        }
                    }
                    return Err(BufferError::Registration {
                        region_id: region.region_id,
                        rollback_failures,
                    });
                }
            }
        }
        Ok(RegisteredRegionTable {
            table: self,
            epoch: RegistrationEpoch::random(),
            handles,
            tracker: TableUseTracker::new(),
        })
    }
}

fn validate_region(
    region: &BufferRegionSpec,
    owner_generation: u64,
    device: u32,
    layout_fingerprint: FixedBytes<32>,
    kv_page_capacity: &mut Option<u64>,
) -> Result<(), BufferError> {
    if region.base_address == 0 || !region.base_address.is_multiple_of(REGION_ALIGNMENT) {
        return invalid("base_address", "must be non-zero and 64-byte aligned");
    }
    if region.length_bytes == 0 {
        return invalid("length_bytes", "must be non-zero");
    }
    region
        .base_address
        .checked_add(region.length_bytes)
        .ok_or(BufferError::InvalidDescriptor {
            field: "base_address",
            detail: "address range overflows u64",
        })?;
    if region.owner_generation != owner_generation {
        return invalid("owner_generation", "does not match the table owner");
    }
    if region.layout_fingerprint != layout_fingerprint {
        return invalid(
            "layout_fingerprint",
            "does not match the frozen table layout",
        );
    }

    match region.region_id {
        0..=27 => validate_kv_region(
            region,
            RegionKind::Key {
                layer: region.region_id,
            },
            device,
            kv_page_capacity,
        ),
        28..=55 => validate_kv_region(
            region,
            RegionKind::Value {
                layer: region.region_id - 28,
            },
            device,
            kv_page_capacity,
        ),
        56 => validate_slot_region(
            region,
            RegionKind::Aux,
            RegionLayout::aux(),
            SLOT_COUNT * AUX_SLOT_BYTES,
        ),
        57 => validate_slot_region(
            region,
            RegionKind::Completion,
            RegionLayout::completion(),
            SLOT_COUNT * COMPLETION_SLOT_BYTES,
        ),
        _ => invalid("region_id", "is outside the frozen mapping"),
    }
}

fn validate_kv_region(
    region: &BufferRegionSpec,
    expected_kind: RegionKind,
    device: u32,
    kv_page_capacity: &mut Option<u64>,
) -> Result<(), BufferError> {
    if region.kind != expected_kind {
        return invalid("kind", "does not match the frozen RegionId mapping");
    }
    if region.location != (RegionLocation::Device { device }) {
        return invalid("location", "KV regions must use the table GPU");
    }
    if region.dtype != BufferDType::BFloat16 {
        return invalid("dtype", "KV regions must use BF16");
    }
    if !region.length_bytes.is_multiple_of(KV_PAGE_BYTES) {
        return invalid("length_bytes", "KV length must contain whole pages");
    }
    let page_capacity = region.length_bytes / KV_PAGE_BYTES;
    if page_capacity == 0 || region.layout != RegionLayout::kv(page_capacity)? {
        return invalid("layout", "KV shape/stride does not match BF16 NHD");
    }
    if kv_page_capacity.is_some_and(|capacity| capacity != page_capacity) {
        return invalid("length_bytes", "all KV regions must have equal capacity");
    }
    *kv_page_capacity = Some(page_capacity);
    Ok(())
}

fn validate_slot_region(
    region: &BufferRegionSpec,
    expected_kind: RegionKind,
    expected_layout: RegionLayout,
    expected_length: u64,
) -> Result<(), BufferError> {
    if region.kind != expected_kind {
        return invalid("kind", "does not match the frozen RegionId mapping");
    }
    if !matches!(
        region.location,
        RegionLocation::PinnedHost { numa_node: 0 | 1 }
    ) {
        return invalid("location", "slot regions must use pinned host memory");
    }
    if region.dtype != BufferDType::Bytes {
        return invalid("dtype", "slot regions must use byte records");
    }
    if region.layout != expected_layout || region.length_bytes != expected_length {
        return invalid("layout", "slot shape/stride or length is not frozen v1");
    }
    Ok(())
}

pub trait RegistrationPort {
    type Handle;

    fn register(
        &mut self,
        region: &BufferRegionSpec,
    ) -> Result<Self::Handle, RegistrationPortError>;

    fn unregister(&mut self, handle: Self::Handle) -> Result<(), RegistrationPortError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationPortError {
    Register,
    Unregister,
    Descriptor,
}

pub struct RegisteredRegionTable<H> {
    table: BufferTable,
    epoch: RegistrationEpoch,
    handles: Vec<Option<H>>,
    tracker: TableUseTracker,
}

impl<H> RegisteredRegionTable<H> {
    pub const fn epoch(&self) -> RegistrationEpoch {
        self.epoch
    }

    pub const fn owner_generation(&self) -> u64 {
        self.table.owner_generation()
    }

    pub const fn kv_page_capacity(&self) -> u64 {
        self.table.kv_page_capacity()
    }

    pub const fn layout_fingerprint(&self) -> FixedBytes<32> {
        self.table.layout_fingerprint()
    }

    pub fn tracker(&self) -> TableUseTracker {
        self.tracker.clone()
    }

    pub fn is_registered(&self) -> bool {
        self.handles.iter().any(Option::is_some)
    }

    pub fn region(&self, region_id: u16) -> Result<&BufferRegionSpec, BufferError> {
        self.table.region(region_id)
    }

    pub(crate) fn registered_handle(&self, region_id: u16) -> Result<&H, BufferError> {
        self.handles
            .get(usize::from(region_id))
            .and_then(Option::as_ref)
            .ok_or(BufferError::StaleRegistration)
    }

    pub fn validate_epoch(&self, epoch: RegistrationEpoch) -> Result<(), BufferError> {
        if epoch != self.epoch {
            return Err(BufferError::StaleRegistration);
        }
        Ok(())
    }

    pub fn resolve_kv_range(
        &self,
        epoch: RegistrationEpoch,
        region_id: u16,
        page: u32,
        offset: u64,
        length: u64,
    ) -> Result<u64, BufferError> {
        self.validate_epoch(epoch)?;
        if region_id >= KV_REGION_COUNT as u16 {
            return invalid("region_id", "KV range must use RegionId 0..=55");
        }
        if u64::from(page) >= self.table.kv_page_capacity {
            return invalid("page", "is outside the registered KV capacity");
        }
        let page_offset =
            u64::from(page)
                .checked_mul(KV_PAGE_BYTES)
                .ok_or(BufferError::InvalidDescriptor {
                    field: "page",
                    detail: "page offset overflows u64",
                })?;
        let range_end = offset
            .checked_add(length)
            .ok_or(BufferError::InvalidDescriptor {
                field: "range",
                detail: "range overflows u64",
            })?;
        if length == 0 || range_end > KV_PAGE_BYTES {
            return invalid("range", "must be non-zero and contained in one KV page");
        }
        let region = self.table.region(region_id)?;
        region
            .base_address
            .checked_add(page_offset)
            .and_then(|address| address.checked_add(offset))
            .ok_or(BufferError::InvalidDescriptor {
                field: "address",
                detail: "resolved address overflows u64",
            })
    }

    pub fn unregister<P>(&mut self, port: &mut P) -> Result<(), BufferError>
    where
        P: RegistrationPort<Handle = H>,
    {
        let use_snapshot = self.tracker.snapshot();
        if use_snapshot.active != 0 || use_snapshot.quarantined != 0 {
            return Err(BufferError::TableInUse {
                active: use_snapshot.active,
                quarantined: use_snapshot.quarantined,
            });
        }
        let mut failures = 0;
        for handle in self.handles.iter_mut().rev().filter_map(Option::take) {
            if port.unregister(handle).is_err() {
                failures += 1;
            }
        }
        if failures == 0 {
            Ok(())
        } else {
            Err(BufferError::Unregistration { failures })
        }
    }

    pub(crate) fn authenticated_region_records(&self) -> Vec<RegionRecord> {
        self.table
            .regions
            .iter()
            .map(|region| RegionRecord {
                region_id: region.region_id,
                remote_base_addr: region.base_address,
                length_bytes: region.length_bytes,
                location: match region.location {
                    RegionLocation::Device { device } => format!("cuda:{device}"),
                    RegionLocation::PinnedHost { numa_node } => format!("cpu:{numa_node}"),
                },
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableUseSnapshot {
    pub active: usize,
    pub quarantined: usize,
}

#[derive(Clone)]
pub struct TableUseTracker {
    inner: Arc<TableUseCounts>,
}

struct TableUseCounts {
    active: AtomicUsize,
    quarantined: AtomicUsize,
}

impl TableUseTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TableUseCounts {
                active: AtomicUsize::new(0),
                quarantined: AtomicUsize::new(0),
            }),
        }
    }

    pub fn acquire(&self) -> TableUseGuard {
        self.inner.active.fetch_add(1, Ordering::SeqCst);
        TableUseGuard {
            tracker: self.clone(),
            state: TableUseState::Active,
        }
    }

    pub fn snapshot(&self) -> TableUseSnapshot {
        TableUseSnapshot {
            active: self.inner.active.load(Ordering::SeqCst),
            quarantined: self.inner.quarantined.load(Ordering::SeqCst),
        }
    }
}

impl Default for TableUseTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableUseState {
    Active,
    Quarantined,
}

pub struct TableUseGuard {
    tracker: TableUseTracker,
    state: TableUseState,
}

impl TableUseGuard {
    pub fn quarantine(&mut self) {
        if self.state == TableUseState::Active {
            self.tracker.inner.active.fetch_sub(1, Ordering::SeqCst);
            self.tracker
                .inner
                .quarantined
                .fetch_add(1, Ordering::SeqCst);
            self.state = TableUseState::Quarantined;
        }
    }
}

impl Drop for TableUseGuard {
    fn drop(&mut self) {
        match self.state {
            TableUseState::Active => {
                self.tracker.inner.active.fetch_sub(1, Ordering::SeqCst);
            }
            TableUseState::Quarantined => {
                self.tracker
                    .inner
                    .quarantined
                    .fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

pub struct MooncakeRegistrationPort<'a> {
    owner: &'a EngineOwner,
    buffers: BTreeMap<u16, MemoryBuffer>,
}

impl<'a> MooncakeRegistrationPort<'a> {
    pub fn new(owner: &'a EngineOwner, buffers: BTreeMap<u16, MemoryBuffer>) -> Self {
        Self { owner, buffers }
    }
}

impl RegistrationPort for MooncakeRegistrationPort<'_> {
    type Handle = MooncakeRegion;

    fn register(
        &mut self,
        region: &BufferRegionSpec,
    ) -> Result<Self::Handle, RegistrationPortError> {
        let buffer = self
            .buffers
            .get(&region.region_id)
            .ok_or(RegistrationPortError::Descriptor)?;
        if buffer.address() != region.base_address || buffer.len() as u64 != region.length_bytes {
            return Err(RegistrationPortError::Descriptor);
        }
        let location = match region.location {
            RegionLocation::Device { device: 4 } => MemoryLocation::Cuda4,
            RegionLocation::Device { device: 5 } => MemoryLocation::Cuda5,
            RegionLocation::PinnedHost { numa_node: 0 } => MemoryLocation::Cpu0,
            RegionLocation::PinnedHost { numa_node: 1 } => MemoryLocation::Cpu1,
            _ => return Err(RegistrationPortError::Descriptor),
        };
        let compatible = matches!(
            (buffer, region.location),
            (
                MemoryBuffer::Cuda(_),
                RegionLocation::Device { device: 4 | 5 }
            ) | (
                MemoryBuffer::Pinned(_),
                RegionLocation::PinnedHost { numa_node: 0 | 1 }
            )
        );
        if !compatible {
            return Err(RegistrationPortError::Descriptor);
        }
        self.owner
            .register_region(buffer.clone(), location)
            .map_err(|_| RegistrationPortError::Register)
    }

    fn unregister(&mut self, handle: Self::Handle) -> Result<(), RegistrationPortError> {
        handle
            .close()
            .map_err(|_| RegistrationPortError::Unregister)
    }
}

fn invalid<T>(field: &'static str, detail: &'static str) -> Result<T, BufferError> {
    Err(BufferError::InvalidDescriptor { field, detail })
}
