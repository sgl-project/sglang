use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use sglang_server::pd::buffer::{
    BufferError, CudaEventQuery, CudaEventRuntime, CudaEventSourceFence, SourceComputeFence,
};
use sglang_server::pd::room::{Clock, ManualClock};

#[derive(Debug, Clone, PartialEq, Eq)]
enum EventCall {
    SetDevice(u32),
    Create,
    Record { event: u64, stream: u64 },
    Query(u64),
    Destroy(u64),
}

struct FakeCuda {
    calls: Arc<Mutex<Vec<EventCall>>>,
    queries: VecDeque<Result<CudaEventQuery, BufferError>>,
    clock: Arc<ManualClock>,
    fail_record: bool,
}

impl CudaEventRuntime for FakeCuda {
    type Event = u64;

    fn set_device(&mut self, device: u32) -> Result<(), BufferError> {
        self.calls
            .lock()
            .expect("calls")
            .push(EventCall::SetDevice(device));
        Ok(())
    }

    fn create_event(&mut self) -> Result<Self::Event, BufferError> {
        self.calls.lock().expect("calls").push(EventCall::Create);
        Ok(17)
    }

    fn record_event(&mut self, event: Self::Event, stream: u64) -> Result<(), BufferError> {
        self.calls
            .lock()
            .expect("calls")
            .push(EventCall::Record { event, stream });
        if self.fail_record {
            Err(BufferError::SourceFence)
        } else {
            Ok(())
        }
    }

    fn query_event(&mut self, event: Self::Event) -> Result<CudaEventQuery, BufferError> {
        self.calls
            .lock()
            .expect("calls")
            .push(EventCall::Query(event));
        self.clock.advance_monotonic(1);
        self.queries
            .pop_front()
            .unwrap_or(Ok(CudaEventQuery::Ready))
    }

    fn destroy_event(&mut self, event: Self::Event) {
        self.calls
            .lock()
            .expect("calls")
            .push(EventCall::Destroy(event));
    }
}

#[test]
fn rust_owns_records_waits_and_destroys_the_cuda_event() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let clock = Arc::new(ManualClock::new(100));
    {
        let runtime = FakeCuda {
            calls: Arc::clone(&calls),
            queries: VecDeque::from([
                Ok(CudaEventQuery::Pending),
                Ok(CudaEventQuery::Pending),
                Ok(CudaEventQuery::Ready),
            ]),
            clock: Arc::clone(&clock),
            fail_record: false,
        };
        let mut fence =
            CudaEventSourceFence::new(5, 0x1234, runtime, Arc::clone(&clock)).expect("fence");
        fence.wait_ready(110).expect("event ready");
    }
    assert_eq!(
        *calls.lock().expect("calls"),
        vec![
            EventCall::SetDevice(5),
            EventCall::Create,
            EventCall::Record {
                event: 17,
                stream: 0x1234,
            },
            EventCall::Query(17),
            EventCall::Query(17),
            EventCall::Query(17),
            EventCall::Destroy(17),
        ]
    );
}

#[test]
fn record_error_and_deadline_fail_closed_and_destroy_once() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let clock = Arc::new(ManualClock::new(100));
    let runtime = FakeCuda {
        calls: Arc::clone(&calls),
        queries: VecDeque::new(),
        clock: Arc::clone(&clock),
        fail_record: true,
    };
    assert!(matches!(
        CudaEventSourceFence::new(4, 9, runtime, Arc::clone(&clock)),
        Err(BufferError::SourceFence)
    ));
    assert_eq!(
        calls
            .lock()
            .expect("calls")
            .iter()
            .filter(|call| matches!(call, EventCall::Destroy(17)))
            .count(),
        1
    );

    let calls = Arc::new(Mutex::new(Vec::new()));
    let runtime = FakeCuda {
        calls: Arc::clone(&calls),
        queries: (0..4).map(|_| Ok(CudaEventQuery::Pending)).collect(),
        clock: Arc::clone(&clock),
        fail_record: false,
    };
    let mut fence = CudaEventSourceFence::new(5, 9, runtime, Arc::clone(&clock)).expect("fence");
    let deadline = clock.now_monotonic_ms() + 2;
    assert_eq!(fence.wait_ready(deadline), Err(BufferError::SourceFence));
    drop(fence);
    assert_eq!(
        calls
            .lock()
            .expect("calls")
            .iter()
            .filter(|call| matches!(call, EventCall::Destroy(17)))
            .count(),
        1
    );
}
