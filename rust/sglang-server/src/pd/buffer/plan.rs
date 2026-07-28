use std::collections::BTreeSet;
use std::fmt;

use sha2::{Digest, Sha256};

use crate::pd::buffer::BufferError;
use crate::pd::buffer::descriptor::{
    KV_PAGE_SIZE_TOKENS, KV_REGION_COUNT, KV_ROW_BYTES, RegisteredRegionTable,
};
use crate::pd::protocol::{KvBlock, PrepareAccepted};
use crate::pd::room::{RegistrationEpoch, RoomId};

const TRANSFER_PLAN_DOMAIN: &[u8] = b"SGLANG-PD-TRANSFER-PLAN-V1\0";
const MAX_KV_PAGES: usize = 64;
const MAX_KV_BLOCKS: usize = KV_REGION_COUNT * MAX_KV_PAGES;
const SLOT_COUNT: u16 = 32;
const MAX_VALID_TOKEN_COUNT: u32 = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPlanInput {
    pub room: RoomId,
    pub transfer_generation: u64,
    pub source_registration_epoch: RegistrationEpoch,
    pub destination_registration_epoch: RegistrationEpoch,
    pub source_pages: Vec<u32>,
    pub destination_pages: Vec<u32>,
    pub source_aux_slot: u16,
    pub destination_aux_slot: u16,
    pub source_completion_slot: u16,
    pub destination_completion_slot: u16,
    pub valid_token_count: u32,
    pub chunk_sequence: u32,
    pub chunk_count: u32,
    pub is_last_chunk: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransferPlanDigest([u8; 32]);

impl TransferPlanDigest {
    pub fn from_hex(value: &str) -> Result<Self, BufferError> {
        let decoded = hex::decode(value).map_err(|_| BufferError::PlanMismatch {
            field: "transfer_plan_digest",
        })?;
        let bytes: [u8; 32] = decoded.try_into().map_err(|_| BufferError::PlanMismatch {
            field: "transfer_plan_digest",
        })?;
        if bytes.iter().all(|byte| *byte == 0) {
            return Err(BufferError::PlanMismatch {
                field: "transfer_plan_digest",
            });
        }
        Ok(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Debug for TransferPlanDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransferPlanDigest")
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPlan {
    input: TransferPlanInput,
    kv_blocks: Vec<KvBlock>,
    canonical_bytes: Vec<u8>,
    digest: TransferPlanDigest,
}

impl TransferPlan {
    pub fn new(input: TransferPlanInput) -> Result<Self, BufferError> {
        validate_input(&input)?;
        let kv_blocks = build_blocks(&input)?;
        if kv_blocks.len() > MAX_KV_BLOCKS {
            return Err(BufferError::PlanLimit { field: "kv_blocks" });
        }
        validate_blocks(&kv_blocks, input.valid_token_count)?;
        let canonical_bytes = canonicalize(&input, &kv_blocks)?;
        let digest = TransferPlanDigest(Sha256::digest(&canonical_bytes).into());
        Ok(Self {
            input,
            kv_blocks,
            canonical_bytes,
            digest,
        })
    }

    pub const fn room(&self) -> RoomId {
        self.input.room
    }

    pub const fn transfer_generation(&self) -> u64 {
        self.input.transfer_generation
    }

    pub const fn source_registration_epoch(&self) -> RegistrationEpoch {
        self.input.source_registration_epoch
    }

    pub const fn destination_registration_epoch(&self) -> RegistrationEpoch {
        self.input.destination_registration_epoch
    }

    pub fn kv_blocks(&self) -> &[KvBlock] {
        &self.kv_blocks
    }

    pub const fn digest(&self) -> TransferPlanDigest {
        self.digest
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub fn expected_kv_bytes(&self) -> u64 {
        self.kv_blocks.iter().map(|block| block.byte_length).sum()
    }

    pub fn source_pages(&self) -> &[u32] {
        &self.input.source_pages
    }

    pub fn destination_pages(&self) -> &[u32] {
        &self.input.destination_pages
    }

    pub const fn source_aux_slot(&self) -> u16 {
        self.input.source_aux_slot
    }

    pub const fn destination_aux_slot(&self) -> u16 {
        self.input.destination_aux_slot
    }

    pub const fn source_completion_slot(&self) -> u16 {
        self.input.source_completion_slot
    }

    pub const fn destination_completion_slot(&self) -> u16 {
        self.input.destination_completion_slot
    }

    pub const fn valid_token_count(&self) -> u32 {
        self.input.valid_token_count
    }

    pub fn verify_destination(
        &self,
        registration_epoch: RegistrationEpoch,
        destination_pages: &[u32],
        aux_slot: u16,
        completion_slot: u16,
    ) -> bool {
        registration_epoch == self.input.destination_registration_epoch
            && destination_pages == self.input.destination_pages
            && aux_slot == self.input.destination_aux_slot
            && completion_slot == self.input.destination_completion_slot
    }

    pub fn verify_prepare_accepted(&self, accepted: &PrepareAccepted) -> Result<(), BufferError> {
        let room = self.input.room;
        if accepted.room.decode_process_epoch.as_bytes() != room.key.decode_process_epoch.as_bytes()
            || accepted.room.bootstrap_room != room.key.bootstrap_room
            || accepted.room.attempt_id.as_bytes() != room.key.attempt_id.as_bytes()
            || accepted.room.generation != room.generation
        {
            return mismatch("room");
        }
        if accepted.source_registration_epoch.as_bytes()
            != self.input.source_registration_epoch.as_bytes()
            || accepted.destination_registration_epoch.as_bytes()
                != self.input.destination_registration_epoch.as_bytes()
        {
            return mismatch("registration_epoch");
        }
        if accepted.kv_blocks != self.kv_blocks {
            return mismatch("kv_blocks");
        }
        if accepted.source_aux_slot != self.input.source_aux_slot
            || accepted.destination_aux_slot != self.input.destination_aux_slot
            || accepted.source_completion_slot != self.input.source_completion_slot
            || accepted.destination_completion_slot != self.input.destination_completion_slot
        {
            return mismatch("slot");
        }
        if accepted.valid_token_count != self.input.valid_token_count {
            return mismatch("valid_token_count");
        }
        if accepted.chunk_sequence != self.input.chunk_sequence
            || accepted.chunk_count != self.input.chunk_count
            || accepted.is_last_chunk != self.input.is_last_chunk
        {
            return mismatch("chunk");
        }
        if accepted.transfer_plan_digest.as_bytes() != self.digest.as_bytes() {
            return mismatch("transfer_plan_digest");
        }
        Ok(())
    }

    pub fn validate_registered_tables<SourceHandle, DestinationHandle>(
        &self,
        source: &RegisteredRegionTable<SourceHandle>,
        destination: &RegisteredRegionTable<DestinationHandle>,
    ) -> Result<(), BufferError> {
        source.validate_epoch(self.input.source_registration_epoch)?;
        destination.validate_epoch(self.input.destination_registration_epoch)?;
        for block in &self.kv_blocks {
            source.resolve_kv_range(
                self.input.source_registration_epoch,
                block.region_id,
                block.source_page,
                block.byte_offset,
                block.byte_length,
            )?;
            destination.resolve_kv_range(
                self.input.destination_registration_epoch,
                block.region_id,
                block.destination_page,
                block.byte_offset,
                block.byte_length,
            )?;
        }
        Ok(())
    }
}

fn validate_input(input: &TransferPlanInput) -> Result<(), BufferError> {
    if input.transfer_generation == 0 {
        return mismatch("transfer_generation");
    }
    let page_count = input.source_pages.len();
    if page_count == 0 || page_count > MAX_KV_PAGES {
        return Err(BufferError::PlanLimit { field: "pages" });
    }
    if input.destination_pages.len() != page_count {
        return mismatch("destination_pages");
    }
    if BTreeSet::from_iter(input.source_pages.iter()).len() != page_count
        || BTreeSet::from_iter(input.destination_pages.iter()).len() != page_count
    {
        return mismatch("page_allocation");
    }
    if !(1..=MAX_VALID_TOKEN_COUNT).contains(&input.valid_token_count) {
        return Err(BufferError::PlanLimit {
            field: "valid_token_count",
        });
    }
    let expected_pages = input.valid_token_count.div_ceil(KV_PAGE_SIZE_TOKENS) as usize;
    if page_count != expected_pages {
        return mismatch("page_count");
    }
    if input.chunk_sequence != 0 || input.chunk_count != 1 || !input.is_last_chunk {
        return mismatch("chunk");
    }
    if [
        input.source_aux_slot,
        input.destination_aux_slot,
        input.source_completion_slot,
        input.destination_completion_slot,
    ]
    .into_iter()
    .any(|slot| slot >= SLOT_COUNT)
    {
        return Err(BufferError::PlanLimit { field: "slot" });
    }
    Ok(())
}

fn build_blocks(input: &TransferPlanInput) -> Result<Vec<KvBlock>, BufferError> {
    let block_count = KV_REGION_COUNT
        .checked_mul(input.source_pages.len())
        .ok_or(BufferError::PlanLimit { field: "kv_blocks" })?;
    let mut blocks = Vec::with_capacity(block_count);
    for region_id in 0..KV_REGION_COUNT as u16 {
        for (logical_page, (&source_page, &destination_page)) in input
            .source_pages
            .iter()
            .zip(&input.destination_pages)
            .enumerate()
        {
            let consumed_tokens = u32::try_from(logical_page)
                .map_err(|_| BufferError::PlanLimit { field: "pages" })?
                .checked_mul(KV_PAGE_SIZE_TOKENS)
                .ok_or(BufferError::PlanLimit { field: "pages" })?;
            let valid_rows = input
                .valid_token_count
                .checked_sub(consumed_tokens)
                .ok_or(BufferError::PlanMismatch {
                    field: "valid_token_count",
                })?
                .min(KV_PAGE_SIZE_TOKENS);
            blocks.push(KvBlock {
                region_id,
                source_page,
                destination_page,
                byte_offset: 0,
                byte_length: u64::from(valid_rows)
                    .checked_mul(KV_ROW_BYTES)
                    .ok_or(BufferError::PlanLimit { field: "range" })?,
            });
        }
    }
    blocks.sort_unstable();
    Ok(blocks)
}

fn validate_blocks(blocks: &[KvBlock], valid_token_count: u32) -> Result<(), BufferError> {
    if blocks.is_empty() || blocks.len() > MAX_KV_BLOCKS {
        return Err(BufferError::PlanLimit { field: "kv_blocks" });
    }
    if blocks.windows(2).any(|pair| pair[0] >= pair[1]) {
        return mismatch("kv_block_order");
    }
    let page_count = valid_token_count.div_ceil(KV_PAGE_SIZE_TOKENS) as usize;
    let expected_per_region = &blocks[..page_count];
    for (region_index, region_blocks) in blocks.chunks_exact(page_count).enumerate() {
        if region_blocks.len() != page_count
            || region_blocks
                .iter()
                .any(|block| usize::from(block.region_id) != region_index)
        {
            return mismatch("region_mapping");
        }
        if region_blocks
            .iter()
            .zip(expected_per_region)
            .any(|(block, expected)| {
                block.source_page != expected.source_page
                    || block.destination_page != expected.destination_page
                    || block.byte_offset != expected.byte_offset
                    || block.byte_length != expected.byte_length
            })
        {
            return mismatch("page_mapping");
        }
    }
    Ok(())
}

fn canonicalize(input: &TransferPlanInput, blocks: &[KvBlock]) -> Result<Vec<u8>, BufferError> {
    let mut canonical = Vec::with_capacity(
        TRANSFER_PLAN_DOMAIN.len() + 16 + 16 + 8 + 4 + blocks.len() * 26 + 8 + 12 + 1,
    );
    canonical.extend_from_slice(TRANSFER_PLAN_DOMAIN);
    canonical.extend_from_slice(&input.source_registration_epoch.as_bytes());
    canonical.extend_from_slice(&input.destination_registration_epoch.as_bytes());
    canonical.extend_from_slice(&input.transfer_generation.to_be_bytes());
    canonical.extend_from_slice(
        &u32::try_from(blocks.len())
            .map_err(|_| BufferError::PlanLimit { field: "kv_blocks" })?
            .to_be_bytes(),
    );
    for block in blocks {
        canonical.extend_from_slice(&block.region_id.to_be_bytes());
        canonical.extend_from_slice(&block.source_page.to_be_bytes());
        canonical.extend_from_slice(&block.destination_page.to_be_bytes());
        canonical.extend_from_slice(&block.byte_offset.to_be_bytes());
        canonical.extend_from_slice(&block.byte_length.to_be_bytes());
    }
    for slot in [
        input.source_aux_slot,
        input.destination_aux_slot,
        input.source_completion_slot,
        input.destination_completion_slot,
    ] {
        canonical.extend_from_slice(&slot.to_be_bytes());
    }
    canonical.extend_from_slice(&input.valid_token_count.to_be_bytes());
    canonical.extend_from_slice(&input.chunk_sequence.to_be_bytes());
    canonical.extend_from_slice(&input.chunk_count.to_be_bytes());
    canonical.push(u8::from(input.is_last_chunk));
    Ok(canonical)
}

fn mismatch<T>(field: &'static str) -> Result<T, BufferError> {
    Err(BufferError::PlanMismatch { field })
}
