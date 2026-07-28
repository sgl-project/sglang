use super::*;

pub(super) struct NativeReceive {
    pub(super) records: CpuRecords,
    pub(super) terminal_kind: u8,
}

pub(super) fn receive_native_data(
    stream: &mut TcpStream,
    plan: &TransferPlan,
    scenario: Scenario,
) -> HarnessResult<NativeReceive> {
    let mut records = CpuRecords::new();
    let mut expected_phase = NativePhase::Kv;
    loop {
        let frame = receive_frame(stream)?;
        match frame.kind {
            KV if expected_phase == NativePhase::Kv => {
                receive_kv(&frame.payload, plan, &mut records)?;
                if scenario == Scenario::TimeoutRecovery {
                    send_frame(stream, NATIVE_PENDING, &[])?;
                    send_frame(stream, NATIVE_SUCCESS, &[])?;
                    let terminal = receive_frame(stream)?;
                    verify_terminal_digest(&terminal, plan)?;
                    return Ok(NativeReceive {
                        records,
                        terminal_kind: terminal.kind,
                    });
                }
                send_frame(stream, NATIVE_SUCCESS, &[])?;
                expected_phase = NativePhase::Aux;
            }
            AUX if expected_phase == NativePhase::Aux => {
                if frame.payload.len() != AUX_BYTES {
                    return Err("aux native operation had the wrong length".into());
                }
                records.aux.copy_from_slice(&frame.payload);
                if scenario == Scenario::SafeFailure {
                    send_frame(stream, NATIVE_FAILURE, &[])?;
                    let terminal = receive_frame(stream)?;
                    verify_terminal_digest(&terminal, plan)?;
                    return Ok(NativeReceive {
                        records,
                        terminal_kind: terminal.kind,
                    });
                }
                send_frame(stream, NATIVE_SUCCESS, &[])?;
                expected_phase = NativePhase::CompletionBody;
            }
            COMPLETION_BODY if expected_phase == NativePhase::CompletionBody => {
                if frame.payload.len() != 188 {
                    return Err("completion body native operation had the wrong length".into());
                }
                records.completion[..188].copy_from_slice(&frame.payload);
                send_frame(stream, NATIVE_SUCCESS, &[])?;
                expected_phase = NativePhase::CompletionMarker;
            }
            COMPLETION_MARKER if expected_phase == NativePhase::CompletionMarker => {
                if frame.payload.as_slice() != b"DONE" {
                    return Err("completion marker was not written last".into());
                }
                records.completion[188..].copy_from_slice(&frame.payload);
                send_frame(stream, NATIVE_SUCCESS, &[])?;
                expected_phase = NativePhase::Kv;
            }
            DATA_READY if expected_phase == NativePhase::Kv => {
                verify_terminal_digest(&frame, plan)?;
                return Ok(NativeReceive {
                    records,
                    terminal_kind: DATA_READY,
                });
            }
            _ => return Err("native stages were duplicated or out of order".into()),
        }
    }
}

pub(super) struct SocketNativePort<'a> {
    stream: &'a mut TcpStream,
    clock: Arc<ManualClock>,
    next_batch: u64,
    expected: HashMap<u64, Vec<u64>>,
}

impl<'a> SocketNativePort<'a> {
    pub(super) fn new(stream: &'a mut TcpStream, clock: Arc<ManualClock>) -> Self {
        Self {
            stream,
            clock,
            next_batch: 1,
            expected: HashMap::new(),
        }
    }
}

impl NativeStagePort for SocketNativePort<'_> {
    fn submit(&mut self, command: &NativeStageCommand) -> Result<NativeBatchToken, BufferError> {
        let (kind, payload) = match command.phase() {
            NativePhase::Kv => (KV, encode_kv(command)?),
            NativePhase::Aux => (AUX, command.payload().to_vec()),
            NativePhase::CompletionBody => (COMPLETION_BODY, command.payload().to_vec()),
            NativePhase::CompletionMarker => (COMPLETION_MARKER, command.payload().to_vec()),
        };
        send_frame(self.stream, kind, &payload).map_err(|_| BufferError::NativeTransfer)?;
        let batch = NativeBatchToken::new(self.next_batch)?;
        self.next_batch = self.next_batch.saturating_add(1);
        self.expected
            .insert(batch.value(), command.expected_lengths().to_vec());
        Ok(batch)
    }

    fn poll(&mut self, batch: NativeBatchToken) -> Result<BatchSnapshot, BufferError> {
        let expected = self
            .expected
            .get(&batch.value())
            .ok_or(BufferError::NativeTransfer)?;
        let frame = receive_frame(self.stream).map_err(|_| BufferError::NativeTransfer)?;
        let (state, safe_terminal) = match frame.kind {
            NATIVE_SUCCESS => (OperationState::Completed, true),
            NATIVE_FAILURE => (OperationState::Failed, true),
            NATIVE_PENDING => {
                self.clock.advance(500);
                (OperationState::Pending, false)
            }
            _ => return Err(BufferError::NativeTransfer),
        };
        Ok(BatchSnapshot {
            operations: expected
                .iter()
                .map(|length| OperationProgress {
                    state,
                    transferred_bytes: if state == OperationState::Completed {
                        *length
                    } else {
                        0
                    },
                })
                .collect(),
            logical_aborted: false,
            safe_terminal,
        })
    }

    fn free_safe(&mut self, batch: NativeBatchToken) -> Result<(), BufferError> {
        self.expected
            .remove(&batch.value())
            .ok_or(BufferError::NativeTransfer)?;
        Ok(())
    }
}

pub(super) struct ReadyFence;

impl SourceComputeFence for ReadyFence {
    fn wait_ready(&mut self, _deadline_monotonic_ms: u64) -> Result<(), BufferError> {
        Ok(())
    }
}

pub(super) struct CpuFlush {
    pub(super) calls: Arc<Mutex<u64>>,
}

impl GpuDirectFlushPort for CpuFlush {
    fn supports_flush_to_owner(&self, device: u32) -> bool {
        device == 5
    }

    fn flush_to_owner(&mut self, device: u32) -> Result<(), BufferError> {
        if device != 5 {
            return Err(BufferError::VisibilityFence);
        }
        let mut calls = self
            .calls
            .lock()
            .map_err(|_| BufferError::VisibilityFence)?;
        *calls = calls.saturating_add(1);
        Ok(())
    }
}

struct PageCopy {
    bytes: Vec<u8>,
    source_page: u32,
}

pub(super) struct CpuRecords {
    aux: [u8; AUX_BYTES],
    completion: [u8; 192],
    pages: HashMap<(u16, u32), PageCopy>,
    pub(super) reads: usize,
    pub(super) clears: usize,
}

impl CpuRecords {
    fn new() -> Self {
        Self {
            aux: [0; AUX_BYTES],
            completion: [0; 192],
            pages: HashMap::new(),
            reads: 0,
            clears: 0,
        }
    }
}

impl DestinationRecordPort for CpuRecords {
    fn read_completion(&mut self, _slot: u16) -> Result<[u8; 192], BufferError> {
        self.reads += 1;
        Ok(self.completion)
    }

    fn read_aux(&mut self, _slot: u16) -> Result<[u8; AUX_BYTES], BufferError> {
        self.reads += 1;
        Ok(self.aux)
    }

    fn clear_final_kv_page_tail(
        &mut self,
        region_id: u16,
        page: u32,
        valid_token_count: u32,
    ) -> Result<(), BufferError> {
        let copied = self
            .pages
            .get_mut(&(region_id, page))
            .ok_or(BufferError::DataRecord { check: "kv_bytes" })?;
        let valid_rows = match valid_token_count % 64 {
            0 => 64,
            remainder => remainder,
        };
        let valid_bytes = usize::try_from(valid_rows)
            .map_err(|_| BufferError::DataRecord { check: "kv_bytes" })?
            * 2_048;
        if !copied.bytes[..valid_bytes]
            .iter()
            .enumerate()
            .all(|(offset, byte)| *byte == data_byte(region_id, copied.source_page, offset))
            || !copied.bytes[valid_bytes..].iter().all(|byte| *byte == 0xa5)
        {
            return Err(BufferError::DataRecord { check: "kv_bytes" });
        }
        clear_partial_page_tail(&mut copied.bytes, valid_token_count)?;
        if copied.bytes[valid_bytes..].iter().any(|byte| *byte != 0) {
            return Err(BufferError::DataRecord {
                check: "partial_page",
            });
        }
        self.clears += 1;
        Ok(())
    }
}

fn receive_kv(payload: &[u8], plan: &TransferPlan, records: &mut CpuRecords) -> HarnessResult<()> {
    let mut reader = BytesReader::new(payload);
    if reader.array::<32>()? != *plan.digest().as_bytes() {
        return Err("KV command used a stale plan digest".into());
    }
    let block_count = usize::try_from(reader.u32()?).map_err(display)?;
    if block_count != plan.kv_blocks().len() {
        return Err("KV command block count did not match the plan".into());
    }
    for expected in plan.kv_blocks() {
        let actual = KvBlock {
            region_id: reader.u16()?,
            source_page: reader.u32()?,
            destination_page: reader.u32()?,
            byte_offset: reader.u64()?,
            byte_length: reader.u64()?,
        };
        if &actual != expected {
            return Err("KV command mutated the accepted plan".into());
        }
        let length = usize::try_from(actual.byte_length).map_err(display)?;
        let bytes = reader.bytes(length)?;
        if !bytes
            .iter()
            .enumerate()
            .all(|(offset, byte)| *byte == data_byte(actual.region_id, actual.source_page, offset))
        {
            return Err("KV command payload did not match source memory".into());
        }
        let start = usize::try_from(actual.byte_offset).map_err(display)?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| "KV command range overflowed".to_string())?;
        if end > KV_PAGE_BYTES {
            return Err("KV command exceeded a page".into());
        }
        let mut page = vec![0xa5; KV_PAGE_BYTES];
        page[start..end].copy_from_slice(bytes);
        if records
            .pages
            .insert(
                (actual.region_id, actual.destination_page),
                PageCopy {
                    bytes: page,
                    source_page: actual.source_page,
                },
            )
            .is_some()
        {
            return Err("KV command repeated a destination range".into());
        }
    }
    reader.finish()
}

fn encode_kv(command: &NativeStageCommand) -> Result<Vec<u8>, BufferError> {
    let byte_capacity = command
        .expected_lengths()
        .iter()
        .try_fold(36_usize, |total, length| {
            total
                .checked_add(26)
                .and_then(|value| value.checked_add(usize::try_from(*length).ok()?))
        })
        .ok_or(BufferError::PlanLimit { field: "kv_bytes" })?;
    let mut payload = Vec::with_capacity(byte_capacity);
    payload.extend_from_slice(command.identity().transfer_plan_digest.as_bytes());
    payload.extend_from_slice(
        &u32::try_from(command.kv_blocks().len())
            .map_err(|_| BufferError::PlanLimit { field: "kv_blocks" })?
            .to_be_bytes(),
    );
    for block in command.kv_blocks() {
        payload.extend_from_slice(&block.region_id.to_be_bytes());
        payload.extend_from_slice(&block.source_page.to_be_bytes());
        payload.extend_from_slice(&block.destination_page.to_be_bytes());
        payload.extend_from_slice(&block.byte_offset.to_be_bytes());
        payload.extend_from_slice(&block.byte_length.to_be_bytes());
        let length = usize::try_from(block.byte_length)
            .map_err(|_| BufferError::PlanLimit { field: "kv_bytes" })?;
        payload.extend(
            (0..length).map(|offset| data_byte(block.region_id, block.source_page, offset)),
        );
    }
    Ok(payload)
}
