use std::sync::{Arc, Mutex};

use sglang_server::mooncake::{BatchSnapshot, OperationProgress, OperationState};
use sglang_server::pd::buffer::{
    BufferError, CudaHostFlushPort, DestinationVisibilityFence, GpuDirectFlushPort, NativeSafety,
    evaluate_native_fence,
};

struct FlushPort {
    supported: bool,
    fail: bool,
    calls: Arc<Mutex<Vec<u32>>>,
}

impl GpuDirectFlushPort for FlushPort {
    fn supports_flush_to_owner(&self, _device: u32) -> bool {
        self.supported
    }

    fn flush_to_owner(&mut self, device: u32) -> Result<(), BufferError> {
        self.calls.lock().expect("flush calls").push(device);
        if self.fail {
            Err(BufferError::VisibilityFence)
        } else {
            Ok(())
        }
    }
}

fn snapshot(
    states: &[(OperationState, u64)],
    logical_aborted: bool,
    safe_terminal: bool,
) -> BatchSnapshot {
    BatchSnapshot {
        operations: states
            .iter()
            .map(|(state, transferred_bytes)| OperationProgress {
                state: *state,
                transferred_bytes: *transferred_bytes,
            })
            .collect(),
        logical_aborted,
        safe_terminal,
    }
}

#[test]
fn native_fence_requires_every_operation_terminal_and_exact_transferred_bytes() {
    assert_eq!(
        evaluate_native_fence(
            &snapshot(
                &[
                    (OperationState::Completed, 64),
                    (OperationState::Pending, 0)
                ],
                false,
                false,
            ),
            &[64, 32],
        ),
        NativeSafety::Pending
    );
    assert_eq!(
        evaluate_native_fence(
            &snapshot(
                &[
                    (OperationState::Completed, 64),
                    (OperationState::Completed, 31)
                ],
                false,
                true,
            ),
            &[64, 32],
        ),
        NativeSafety::SafeFailure
    );
    assert_eq!(
        evaluate_native_fence(
            &snapshot(
                &[(OperationState::Completed, 64), (OperationState::Failed, 0)],
                false,
                true,
            ),
            &[64, 32],
        ),
        NativeSafety::SafeFailure
    );
    assert_eq!(
        evaluate_native_fence(
            &snapshot(
                &[
                    (OperationState::Completed, 64),
                    (OperationState::Completed, 32)
                ],
                true,
                true,
            ),
            &[64, 32],
        ),
        NativeSafety::SafeSuccess,
        "logical abort cannot replace or invalidate an observed native terminal"
    );
}

#[test]
fn destination_visibility_capability_device_and_flush_errors_fail_closed() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    assert!(matches!(
        DestinationVisibilityFence::new(
            5,
            FlushPort {
                supported: false,
                fail: false,
                calls: Arc::clone(&calls),
            }
        ),
        Err(BufferError::VisibilityFence)
    ));
    assert!(matches!(
        DestinationVisibilityFence::new(
            3,
            FlushPort {
                supported: true,
                fail: false,
                calls: Arc::clone(&calls),
            }
        ),
        Err(BufferError::VisibilityFence)
    ));
    assert!(calls.lock().expect("calls").is_empty());

    let mut fence = DestinationVisibilityFence::new(
        5,
        FlushPort {
            supported: true,
            fail: true,
            calls: Arc::clone(&calls),
        },
    )
    .expect("supported visibility fence");
    assert!(matches!(fence.flush(), Err(BufferError::VisibilityFence)));
    assert_eq!(*calls.lock().expect("calls"), vec![5]);
}

#[test]
fn production_cuda_visibility_loader_rejects_missing_library_and_symbols() {
    assert!(matches!(
        CudaHostFlushPort::load("/definitely/missing/libcudart.so"),
        Err(BufferError::VisibilityFence)
    ));
    let libc = [
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/usr/lib/x86_64-linux-gnu/libc.so.6",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).is_file())
    .expect("test host provides libc");
    assert!(matches!(
        CudaHostFlushPort::load(libc),
        Err(BufferError::VisibilityFence)
    ));
}
