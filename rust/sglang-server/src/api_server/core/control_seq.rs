//! Numbers for the control requests every attention-DP rank's scheduler
//! receives. Each DP group's scheduler receives a control request in its own
//! step, so a collective the request runs across groups would meet another
//! group's per-step collectives. The schedulers hold a numbered request until every
//! group has received it and then apply it in the same step, which needs one
//! numbering: rank 0's listener gives the numbers, whichever listener a request
//! arrives at, and every listener hands numbered requests to its scheduler in
//! number order.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::watch;

/// The route rank 0's listener gives out the next number on.
pub(crate) const CONTROL_SEQ_PATH: &str = "/internal/control_seq";

pub(crate) struct ControlSequencer {
    /// The last number given out; used on rank 0's listener only.
    last_issued: AtomicU64,
    /// The number of the next request this listener hands to its scheduler.
    next_submit: watch::Sender<u64>,
}

impl Default for ControlSequencer {
    fn default() -> Self {
        Self {
            last_issued: AtomicU64::new(0),
            next_submit: watch::Sender::new(1),
        }
    }
}

impl ControlSequencer {
    /// The next number, on rank 0's listener.
    pub(crate) fn issue(&self) -> u64 {
        self.last_issued.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Run `submit`, which hands request `seq` to the scheduler, after every
    /// lower-numbered request was handed over: requests reach the listeners in
    /// any order, the scheduler must see them in number order.
    pub(crate) async fn submit_in_order<T>(&self, seq: u64, submit: impl Future<Output = T>) -> T {
        let mut next = self.next_submit.subscribe();
        // The sender lives in `self`, so the wait cannot fail.
        let _ = next.wait_for(|next| *next == seq).await;
        // Passes the turn on even if this future is dropped mid-submit.
        let _turn = Turn {
            sequencer: self,
            done: seq,
        };
        submit.await
    }
}

struct Turn<'a> {
    sequencer: &'a ControlSequencer,
    done: u64,
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.sequencer.next_submit.send_replace(self.done + 1);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn issue_counts_from_one() {
        let seq = ControlSequencer::default();
        assert_eq!((seq.issue(), seq.issue(), seq.issue()), (1, 2, 3));
    }

    /// Requests that arrive out of order are handed over in number order.
    #[tokio::test]
    async fn submit_in_order_reorders() {
        let seq = Arc::new(ControlSequencer::default());
        let order = Arc::new(Mutex::new(Vec::new()));
        let submit = |n: u64| {
            let (seq, order) = (seq.clone(), order.clone());
            tokio::spawn(async move {
                seq.submit_in_order(n, async { order.lock().unwrap().push(n) })
                    .await
            })
        };
        let late = [submit(3), submit(2)];
        tokio::task::yield_now().await;
        assert!(order.lock().unwrap().is_empty());
        submit(1).await.unwrap();
        for task in late {
            task.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }
}
