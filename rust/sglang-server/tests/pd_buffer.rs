use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use serde::Deserialize;
use sglang_server::pd::buffer::{
    BufferDType, BufferError, BufferRegionSpec, BufferTable, CapacityLedger, RegionKind,
    RegionLayout, RegionLocation, RegistrationPort, RegistrationPortError, ReservationRequest,
    TableUseTracker, TransferPlan, TransferPlanInput, TransferStage, TransitionResult,
};
use sglang_server::pd::config::PdProfileV1;
use sglang_server::pd::protocol::{FixedBytes, PrepareAccepted, RoomFields};
use sglang_server::pd::room::{AttemptId, ProcessEpoch, RegistrationEpoch, RoomId, RoomKey};

const DATA_GOLDEN: &[u8] = include_bytes!("../contracts/data-v1-golden.json");
const OWNER_GENERATION: u64 = 11;
const DEVICE: u32 = 5;

#[derive(Debug, Deserialize)]
struct DataGolden {
    transfer_plans: Vec<PlanGolden>,
}

#[derive(Debug, Deserialize)]
struct PlanGolden {
    name: String,
    source_registration_epoch: String,
    destination_registration_epoch: String,
    transfer_generation: u64,
    block_pattern: BlockPattern,
    source_aux_slot: u16,
    destination_aux_slot: u16,
    source_completion_slot: u16,
    destination_completion_slot: u16,
    valid_token_count: u32,
    chunk_sequence: u32,
    chunk_count: u32,
    is_last_chunk: bool,
    expected_block_count: usize,
    expected_canonical_bytes: usize,
    expected_digest_hex: String,
}

#[derive(Debug, Deserialize)]
struct BlockPattern {
    page_count: u32,
    physical_page_modulus: u32,
    source_page_multiplier: u32,
    source_page_offset: u32,
    destination_page_multiplier: u32,
    destination_page_offset: u32,
}

#[derive(Default)]
struct MockRegistrationPort {
    fail_register_at: Option<u16>,
    fail_unregister_at: Option<u16>,
    events: Arc<Mutex<Vec<(bool, u16)>>>,
}

impl RegistrationPort for MockRegistrationPort {
    type Handle = u16;

    fn register(
        &mut self,
        region: &BufferRegionSpec,
    ) -> Result<Self::Handle, RegistrationPortError> {
        if self.fail_register_at == Some(region.region_id) {
            return Err(RegistrationPortError::Register);
        }
        self.events
            .lock()
            .expect("registration events")
            .push((true, region.region_id));
        Ok(region.region_id)
    }

    fn unregister(&mut self, handle: Self::Handle) -> Result<(), RegistrationPortError> {
        self.events
            .lock()
            .expect("registration events")
            .push((false, handle));
        if self.fail_unregister_at == Some(handle) {
            Err(RegistrationPortError::Unregister)
        } else {
            Ok(())
        }
    }
}

fn fingerprint() -> FixedBytes<32> {
    FixedBytes::new([0x33; 32])
}

fn valid_specs(kv_pages: u64) -> Vec<BufferRegionSpec> {
    (0_u16..58)
        .map(|region_id| {
            let (kind, location, dtype, layout, length_bytes) = match region_id {
                0..=27 => (
                    RegionKind::Key { layer: region_id },
                    RegionLocation::Device { device: DEVICE },
                    BufferDType::BFloat16,
                    RegionLayout::kv(kv_pages).expect("KV layout"),
                    kv_pages * 131_072,
                ),
                28..=55 => (
                    RegionKind::Value {
                        layer: region_id - 28,
                    },
                    RegionLocation::Device { device: DEVICE },
                    BufferDType::BFloat16,
                    RegionLayout::kv(kv_pages).expect("KV layout"),
                    kv_pages * 131_072,
                ),
                56 => (
                    RegionKind::Aux,
                    RegionLocation::PinnedHost { numa_node: 0 },
                    BufferDType::Bytes,
                    RegionLayout::aux(),
                    32 * 64,
                ),
                57 => (
                    RegionKind::Completion,
                    RegionLocation::PinnedHost { numa_node: 0 },
                    BufferDType::Bytes,
                    RegionLayout::completion(),
                    32 * 192,
                ),
                _ => unreachable!(),
            };
            BufferRegionSpec {
                region_id,
                kind,
                base_address: 0x1000_0000 + u64::from(region_id) * 0x0100_0000,
                length_bytes,
                location,
                dtype,
                layout,
                owner_generation: OWNER_GENERATION,
                layout_fingerprint: fingerprint(),
            }
        })
        .collect()
}

fn valid_table(kv_pages: u64) -> BufferTable {
    BufferTable::new(
        valid_specs(kv_pages),
        OWNER_GENERATION,
        DEVICE,
        fingerprint(),
    )
    .expect("valid frozen BufferTable")
}

fn room(seed: u64) -> RoomId {
    RoomId::new(
        RoomKey::new(ProcessEpoch::random(), seed, AttemptId::random()).expect("RoomKey"),
        1,
    )
    .expect("RoomId")
}

fn reservation(seed: u16) -> ReservationRequest {
    ReservationRequest {
        room: room(u64::from(seed)),
        handle_generation: 1,
        source_pages: vec![u32::from(seed)],
        destination_pages: vec![u32::from(seed)],
        aux_slot: seed,
        completion_slot: seed,
        request_slot: seed,
        kv_bytes: 56 * 131_072,
        deadline_monotonic_ms: 60_000,
    }
}

#[test]
fn complete_buffer_table_accepts_only_the_frozen_58_region_layout() {
    let table = valid_table(64);
    assert_eq!(table.regions().len(), 58);
    assert_eq!(table.kv_page_capacity(), 64);

    let mut cases = Vec::new();

    let mut missing = valid_specs(64);
    missing.pop();
    cases.push(missing);

    let mut duplicate = valid_specs(64);
    duplicate[1].region_id = 0;
    cases.push(duplicate);

    let mut wrong_order = valid_specs(64);
    wrong_order.swap(0, 1);
    cases.push(wrong_order);

    let mut wrong_kind = valid_specs(64);
    wrong_kind[0].kind = RegionKind::Value { layer: 0 };
    cases.push(wrong_kind);

    let mut wrong_location = valid_specs(64);
    wrong_location[0].location = RegionLocation::PinnedHost { numa_node: 0 };
    cases.push(wrong_location);

    let mut wrong_dtype = valid_specs(64);
    wrong_dtype[0].dtype = BufferDType::Bytes;
    cases.push(wrong_dtype);

    let mut wrong_stride = valid_specs(64);
    wrong_stride[0].layout.strides_bytes[1] = 1024;
    cases.push(wrong_stride);

    let mut wrong_shape = valid_specs(64);
    wrong_shape[0].layout.shape[2] = 7;
    cases.push(wrong_shape);

    let mut wrong_page_size = valid_specs(64);
    wrong_page_size[0].layout.page_size_tokens = 32;
    cases.push(wrong_page_size);

    let mut wrong_item_size = valid_specs(64);
    wrong_item_size[0].layout.item_size_bytes = 4;
    cases.push(wrong_item_size);

    let mut zero_address = valid_specs(64);
    zero_address[0].base_address = 0;
    cases.push(zero_address);

    let mut misaligned = valid_specs(64);
    misaligned[0].base_address += 1;
    cases.push(misaligned);

    let mut zero_length = valid_specs(64);
    zero_length[0].length_bytes = 0;
    cases.push(zero_length);

    let mut overflow = valid_specs(64);
    overflow[0].base_address = u64::MAX - 63;
    cases.push(overflow);

    let mut inconsistent_capacity = valid_specs(64);
    inconsistent_capacity[1].length_bytes -= 131_072;
    inconsistent_capacity[1].layout =
        RegionLayout::kv(63).expect("mutated layout remains internally valid");
    cases.push(inconsistent_capacity);

    let mut stale_generation = valid_specs(64);
    stale_generation[0].owner_generation -= 1;
    cases.push(stale_generation);

    let mut stale_fingerprint = valid_specs(64);
    stale_fingerprint[0].layout_fingerprint = FixedBytes::new([0x44; 32]);
    cases.push(stale_fingerprint);

    let mut partial_page = valid_specs(64);
    partial_page[0].length_bytes -= 1;
    cases.push(partial_page);

    let mut wrong_aux_location = valid_specs(64);
    wrong_aux_location[56].location = RegionLocation::Device { device: DEVICE };
    cases.push(wrong_aux_location);

    let mut wrong_aux_layout = valid_specs(64);
    wrong_aux_layout[56].layout.shape[0] = 31;
    cases.push(wrong_aux_layout);

    let mut wrong_completion_length = valid_specs(64);
    wrong_completion_length[57].length_bytes -= 64;
    cases.push(wrong_completion_length);

    for specs in cases {
        assert!(
            BufferTable::new(specs, OWNER_GENERATION, DEVICE, fingerprint()).is_err(),
            "invalid BufferTable mutation unexpectedly loaded"
        );
    }
    assert!(BufferTable::new(valid_specs(64), 0, DEVICE, fingerprint()).is_err());
    assert!(BufferTable::new(valid_specs(64), OWNER_GENERATION, 3, fingerprint()).is_err());
    assert!(
        BufferTable::new(
            valid_specs(64),
            OWNER_GENERATION,
            DEVICE,
            FixedBytes::new([0; 32]),
        )
        .is_err()
    );
}

#[test]
fn registration_rolls_back_in_reverse_and_active_or_quarantined_use_blocks_unregister() {
    for fail_at in 0_u16..58 {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut failing = MockRegistrationPort {
            fail_register_at: Some(fail_at),
            events: Arc::clone(&events),
            ..MockRegistrationPort::default()
        };
        assert!(matches!(
            valid_table(64).register(&mut failing),
            Err(BufferError::Registration { region_id, rollback_failures: 0 })
                if region_id == fail_at
        ));
        let events = events.lock().expect("events").clone();
        assert_eq!(
            events,
            (0_u16..fail_at)
                .map(|region| (true, region))
                .chain((0_u16..fail_at).rev().map(|region| (false, region)))
                .collect::<Vec<_>>(),
            "registration failure at RegionId {fail_at} did not roll back once in reverse"
        );
    }

    let mut rollback_failure = MockRegistrationPort {
        fail_register_at: Some(5),
        fail_unregister_at: Some(2),
        ..MockRegistrationPort::default()
    };
    assert!(matches!(
        valid_table(64).register(&mut rollback_failure),
        Err(BufferError::Registration {
            region_id: 5,
            rollback_failures: 1
        })
    ));

    let mut port = MockRegistrationPort::default();
    let mut registered = valid_table(64).register(&mut port).expect("register table");
    let epoch = registered.epoch();
    let mut active = registered.tracker().acquire();
    assert!(matches!(
        registered.unregister(&mut port),
        Err(BufferError::TableInUse { .. })
    ));
    active.quarantine();
    assert!(matches!(
        registered.unregister(&mut port),
        Err(BufferError::TableInUse { .. })
    ));
    drop(active);
    registered.unregister(&mut port).expect("safe unregister");
    assert!(!registered.is_registered());
    assert_eq!(registered.epoch(), epoch);
}

#[test]
fn transfer_plan_matches_external_golden_and_rejects_all_boundary_mutations() {
    let golden: DataGolden = serde_json::from_slice(DATA_GOLDEN).expect("data golden");
    for fixture in golden.transfer_plans {
        let input = plan_input(&fixture);
        let plan = TransferPlan::new(input.clone()).expect("golden plan");
        assert_eq!(plan.kv_blocks().len(), fixture.expected_block_count);
        assert_eq!(
            plan.canonical_bytes().len(),
            fixture.expected_canonical_bytes
        );
        assert_eq!(plan.digest().to_hex(), fixture.expected_digest_hex);
        assert!(plan.verify_destination(
            input.destination_registration_epoch,
            &input.destination_pages,
            input.destination_aux_slot,
            input.destination_completion_slot,
        ));

        let mut mutated_pages = input.destination_pages.clone();
        if mutated_pages.len() > 1 {
            mutated_pages.swap(0, 1);
            assert!(!plan.verify_destination(
                input.destination_registration_epoch,
                &mutated_pages,
                input.destination_aux_slot,
                input.destination_completion_slot,
            ));
        }
    }

    let maximum = plan_fixture("maximum_fragmented");
    let mut too_many_pages = plan_input(maximum);
    too_many_pages.source_pages.push(64);
    too_many_pages.destination_pages.push(64);
    too_many_pages.valid_token_count = 4096;
    assert!(matches!(
        TransferPlan::new(too_many_pages),
        Err(BufferError::PlanLimit { .. })
    ));

    let mut stale_epoch = plan_input(maximum);
    stale_epoch.source_registration_epoch = RegistrationEpoch::random();
    let plan = TransferPlan::new(stale_epoch).expect("structurally valid stale plan");
    let mut source_port = MockRegistrationPort::default();
    let mut destination_port = MockRegistrationPort::default();
    let source = valid_table(64)
        .register(&mut source_port)
        .expect("source registration");
    let destination = valid_table(64)
        .register(&mut destination_port)
        .expect("destination registration");
    assert!(matches!(
        plan.validate_registered_tables(&source, &destination),
        Err(BufferError::StaleRegistration)
    ));

    let mut out_of_range = plan_input(plan_fixture("one_page"));
    out_of_range.source_registration_epoch = source.epoch();
    out_of_range.destination_registration_epoch = destination.epoch();
    out_of_range.source_pages[0] = 64;
    let out_of_range = TransferPlan::new(out_of_range).expect("structurally valid page");
    assert!(matches!(
        out_of_range.validate_registered_tables(&source, &destination),
        Err(BufferError::InvalidDescriptor { field: "page", .. })
    ));
}

#[test]
fn destination_rebuild_rejects_every_wire_plan_identity_and_block_mutation() {
    let input = plan_input(plan_fixture("fragmented_partial_page"));
    let plan = TransferPlan::new(input.clone()).expect("plan");
    let accepted = PrepareAccepted {
        room: RoomFields {
            decode_process_epoch: FixedBytes::new(input.room.key.decode_process_epoch.as_bytes()),
            bootstrap_room: input.room.key.bootstrap_room,
            attempt_id: FixedBytes::new(input.room.key.attempt_id.as_bytes()),
            generation: input.room.generation,
            request_contract_digest: FixedBytes::new([0xa1; 32]),
        },
        source_registration_epoch: FixedBytes::new(input.source_registration_epoch.as_bytes()),
        destination_registration_epoch: FixedBytes::new(
            input.destination_registration_epoch.as_bytes(),
        ),
        kv_blocks: plan.kv_blocks().to_vec(),
        source_aux_slot: input.source_aux_slot,
        destination_aux_slot: input.destination_aux_slot,
        source_completion_slot: input.source_completion_slot,
        destination_completion_slot: input.destination_completion_slot,
        valid_token_count: input.valid_token_count,
        chunk_sequence: input.chunk_sequence,
        chunk_count: input.chunk_count,
        is_last_chunk: input.is_last_chunk,
        transfer_plan_digest: FixedBytes::new(*plan.digest().as_bytes()),
    };
    plan.verify_prepare_accepted(&accepted)
        .expect("matching accepted plan");

    let mut mutations = Vec::new();
    let mut mutated = accepted.clone();
    mutated.room.bootstrap_room += 1;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.source_registration_epoch = FixedBytes::new(RegistrationEpoch::random().as_bytes());
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.destination_registration_epoch =
        FixedBytes::new(RegistrationEpoch::random().as_bytes());
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.kv_blocks[0].source_page += 1;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.kv_blocks[0].destination_page += 1;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.kv_blocks[0].region_id += 1;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.kv_blocks[0].byte_offset += 2_048;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.kv_blocks[0].byte_length -= 2_048;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.kv_blocks.swap(0, 1);
    mutations.push(mutated);
    for slot_field in 0..4 {
        let mut mutated = accepted.clone();
        match slot_field {
            0 => mutated.source_aux_slot += 1,
            1 => mutated.destination_aux_slot += 1,
            2 => mutated.source_completion_slot += 1,
            3 => mutated.destination_completion_slot += 1,
            _ => unreachable!(),
        }
        mutations.push(mutated);
    }
    let mut mutated = accepted.clone();
    mutated.valid_token_count += 1;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.chunk_sequence += 1;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.chunk_count += 1;
    mutations.push(mutated);
    let mut mutated = accepted.clone();
    mutated.is_last_chunk = false;
    mutations.push(mutated);
    let mut mutated = accepted;
    mutated.transfer_plan_digest = FixedBytes::new([0xff; 32]);
    mutations.push(mutated);

    for mutation in mutations {
        assert!(
            plan.verify_prepare_accepted(&mutation).is_err(),
            "wire plan mutation unexpectedly matched the destination rebuild"
        );
    }

    let mutators: [fn(&mut TransferPlanInput); 5] = [
        |input: &mut TransferPlanInput| input.source_pages[0] += 1,
        |input: &mut TransferPlanInput| input.destination_pages[0] += 1,
        |input: &mut TransferPlanInput| input.transfer_generation += 1,
        |input: &mut TransferPlanInput| input.valid_token_count += 1,
        |input: &mut TransferPlanInput| input.source_aux_slot += 1,
    ];
    for mutate in mutators {
        let mut mutated = input.clone();
        mutate(&mut mutated);
        let mutated = TransferPlan::new(mutated).expect("valid alternative plan");
        assert_ne!(mutated.digest(), plan.digest());
    }

    let mut duplicate = input.clone();
    duplicate.source_pages[1] = duplicate.source_pages[0];
    assert!(TransferPlan::new(duplicate).is_err());
    let mut invalid_token = input.clone();
    invalid_token.valid_token_count = 0;
    assert!(TransferPlan::new(invalid_token).is_err());
    let mut invalid_chunk = input;
    invalid_chunk.chunk_count = 2;
    assert!(TransferPlan::new(invalid_chunk).is_err());
}

#[test]
fn token_boundaries_produce_exact_page_counts_and_only_valid_final_rows() {
    for valid_token_count in [1_u32, 63, 64, 65, 4_096] {
        let page_count = valid_token_count.div_ceil(64);
        let pages = (0..page_count).collect::<Vec<_>>();
        let plan = TransferPlan::new(TransferPlanInput {
            room: room(u64::from(valid_token_count)),
            transfer_generation: 1,
            source_registration_epoch: RegistrationEpoch::random(),
            destination_registration_epoch: RegistrationEpoch::random(),
            source_pages: pages.clone(),
            destination_pages: pages,
            source_aux_slot: 1,
            destination_aux_slot: 1,
            source_completion_slot: 1,
            destination_completion_slot: 1,
            valid_token_count,
            chunk_sequence: 0,
            chunk_count: 1,
            is_last_chunk: true,
        })
        .expect("boundary plan");
        assert_eq!(plan.kv_blocks().len(), 56 * page_count as usize);
        let final_rows = match valid_token_count % 64 {
            0 => 64,
            remainder => remainder,
        };
        for region_blocks in plan.kv_blocks().chunks_exact(page_count as usize) {
            assert!(
                region_blocks[..region_blocks.len() - 1]
                    .iter()
                    .all(|block| block.byte_length == 131_072)
            );
            assert_eq!(
                region_blocks.last().expect("final block").byte_length,
                u64::from(final_rows) * 2_048
            );
        }
    }
}

#[test]
fn concurrent_capacity_reservation_is_atomic_and_rolls_back_to_baseline() {
    let profile = PdProfileV1::load_embedded().expect("profile");
    let source_tracker = TableUseTracker::new();
    let destination_tracker = TableUseTracker::new();
    let ledger = Arc::new(CapacityLedger::new(
        &profile,
        source_tracker.clone(),
        destination_tracker.clone(),
    ));
    let barrier = Arc::new(Barrier::new(40));
    let handles = (0_u16..40)
        .map(|seed| {
            let ledger = Arc::clone(&ledger);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                ledger.reserve(reservation(seed % 32))
            })
        })
        .collect::<Vec<_>>();

    let mut reserved = Vec::new();
    for result in handles {
        if let Ok(handle) = result.join().expect("reservation thread") {
            reserved.push(handle);
        }
    }
    assert_eq!(reserved.len(), 32);
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.active_rooms, 32);
    assert_eq!(snapshot.source_kv_pages, 32);
    assert_eq!(snapshot.destination_kv_pages, 32);
    assert_eq!(snapshot.aux_slots, 32);
    assert_eq!(snapshot.request_slots, 32);

    for handle in reserved {
        assert_eq!(
            ledger.abort_pre_submit(handle).expect("abort reservation"),
            TransitionResult::Applied
        );
    }
    assert_eq!(ledger.snapshot().active_rooms, 0);
    assert_eq!(source_tracker.snapshot().active, 0);
    assert_eq!(destination_tracker.snapshot().active, 0);
}

#[test]
fn every_frozen_capacity_limit_is_enforced_and_returns_to_baseline() {
    let profile = PdProfileV1::load_embedded().expect("profile");
    let ledger = CapacityLedger::new(&profile, TableUseTracker::new(), TableUseTracker::new());
    let mut handles = Vec::new();
    for seed in 0_u16..32 {
        let first_page = u32::from(seed) * 64;
        let pages = (first_page..first_page + 64).collect::<Vec<_>>();
        let mut request = reservation(seed);
        request.source_pages = pages.clone();
        request.destination_pages = pages;
        request.kv_bytes = 1;
        handles.push(ledger.reserve(request).expect("64-page room reservation"));
    }
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.active_rooms, 32);
    assert_eq!(snapshot.source_kv_pages, 2_048);
    assert_eq!(snapshot.destination_kv_pages, 2_048);
    assert_eq!(snapshot.aux_slots, 32);
    assert_eq!(snapshot.completion_slots, 32);
    assert_eq!(snapshot.request_slots, 32);

    for handle in &handles[..4] {
        ledger
            .begin_stage(*handle, TransferStage::Kv)
            .expect("four native transfers");
    }
    assert!(matches!(
        ledger.begin_stage(handles[4], TransferStage::Kv),
        Err(BufferError::CapacityExhausted {
            resource: "native_transfers"
        })
    ));
    assert_eq!(ledger.snapshot().in_flight_transfers, 4);
    for handle in &handles[..4] {
        ledger
            .finish_stage(*handle, TransferStage::Kv)
            .expect("finish transfer");
        ledger
            .release_failed_safe(*handle)
            .expect("safe failure cleanup");
    }
    for handle in &handles[4..] {
        ledger
            .abort_pre_submit(*handle)
            .expect("pre-submit cleanup");
    }
    assert_eq!(ledger.snapshot().active_rooms, 0);

    let pending = CapacityLedger::new(&profile, TableUseTracker::new(), TableUseTracker::new());
    let mut full = reservation(0);
    full.kv_bytes = profile.capacity.pending_transfer_bytes_per_pair;
    let full_handle = pending.reserve(full).expect("exact pending byte limit");
    let mut one_more = reservation(1);
    one_more.kv_bytes = 1;
    assert!(matches!(
        pending.reserve(one_more),
        Err(BufferError::CapacityExhausted {
            resource: "pending_bytes"
        })
    ));
    pending
        .abort_pre_submit(full_handle)
        .expect("pending byte cleanup");
    assert_eq!(pending.snapshot().pending_bytes, 0);

    let mut oversized = reservation(0);
    oversized.source_pages = (0..65).collect();
    oversized.destination_pages = (0..65).collect();
    assert!(matches!(
        pending.reserve(oversized),
        Err(BufferError::PlanLimit { field: "pages" })
    ));
    let mut invalid_slot = reservation(0);
    invalid_slot.request_slot = 32;
    assert!(matches!(
        pending.reserve(invalid_slot),
        Err(BufferError::PlanLimit { field: "slot" })
    ));
    let mut too_many_bytes = reservation(0);
    too_many_bytes.kv_bytes = profile
        .capacity
        .pending_transfer_bytes_per_pair
        .saturating_add(1);
    assert!(matches!(
        pending.reserve(too_many_bytes),
        Err(BufferError::CapacityExhausted {
            resource: "pending_bytes"
        })
    ));
}

#[path = "pd_buffer/lease.rs"]
mod lease;

fn plan_fixture<'a>(name: &str) -> &'a PlanGolden {
    let golden = Box::leak(Box::new(
        serde_json::from_slice::<DataGolden>(DATA_GOLDEN).expect("data golden"),
    ));
    golden
        .transfer_plans
        .iter()
        .find(|fixture| fixture.name == name)
        .expect("named plan fixture")
}

fn plan_input(fixture: &PlanGolden) -> TransferPlanInput {
    let source_pages = (0..fixture.block_pattern.page_count)
        .map(|page| {
            (page * fixture.block_pattern.source_page_multiplier
                + fixture.block_pattern.source_page_offset)
                % fixture.block_pattern.physical_page_modulus
        })
        .collect();
    let destination_pages = (0..fixture.block_pattern.page_count)
        .map(|page| {
            (page * fixture.block_pattern.destination_page_multiplier
                + fixture.block_pattern.destination_page_offset)
                % fixture.block_pattern.physical_page_modulus
        })
        .collect();
    TransferPlanInput {
        room: room(0),
        transfer_generation: fixture.transfer_generation,
        source_registration_epoch: RegistrationEpoch::parse(&fixture.source_registration_epoch)
            .expect("source epoch"),
        destination_registration_epoch: RegistrationEpoch::parse(
            &fixture.destination_registration_epoch,
        )
        .expect("destination epoch"),
        source_pages,
        destination_pages,
        source_aux_slot: fixture.source_aux_slot,
        destination_aux_slot: fixture.destination_aux_slot,
        source_completion_slot: fixture.source_completion_slot,
        destination_completion_slot: fixture.destination_completion_slot,
        valid_token_count: fixture.valid_token_count,
        chunk_sequence: fixture.chunk_sequence,
        chunk_count: fixture.chunk_count,
        is_last_chunk: fixture.is_last_chunk,
    }
}
