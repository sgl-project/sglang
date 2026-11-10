//! Bounded pure demand/job bookkeeping. Exact transfer specifications coalesce;
//! task interests and physical cancellation/drain are separate lifetimes.
use crate::models::SessionIdentity;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DemandId {
    pub task: SessionIdentity,
    pub id: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BookError {
    Capacity,
    Conflict,
    Unknown,
}
#[derive(Clone, Debug)]
pub struct JobInterest {
    pub interests: BTreeSet<DemandId>,
    pub cancel_requested: bool,
    pub quiescent: bool,
}
pub struct DemandBook {
    max_intents: usize,
    max_jobs: usize,
    demands: BTreeMap<DemandId, BTreeSet<String>>,
    jobs: BTreeMap<String, JobInterest>,
}
impl DemandBook {
    pub fn new(max_intents: usize, max_jobs: usize) -> Self {
        Self {
            max_intents,
            max_jobs,
            demands: BTreeMap::new(),
            jobs: BTreeMap::new(),
        }
    }
    /// Caller supplies a collision-resistant digest of the exact immutable spec,
    /// including action, compatibility, prefix, destination owner/epoch and tier.
    /// Admit all children atomically. A draining job cannot take new interests.
    pub fn attach(&mut self, demand: DemandId, jobs: BTreeSet<String>) -> Result<(), BookError> {
        if let Some(old) = self.demands.get(&demand) {
            return if old == &jobs {
                Ok(())
            } else {
                Err(BookError::Conflict)
            };
        }
        if self.demands.len() >= self.max_intents
            || self.jobs.len()
                + jobs
                    .iter()
                    .filter(|id| !self.jobs.contains_key(*id))
                    .count()
                > self.max_jobs
        {
            return Err(BookError::Capacity);
        }
        if jobs
            .iter()
            .any(|id| self.jobs.get(id).is_some_and(|j| j.cancel_requested))
        {
            return Err(BookError::Conflict);
        }
        for id in &jobs {
            self.jobs
                .entry(id.clone())
                .or_insert_with(|| JobInterest {
                    interests: BTreeSet::new(),
                    cancel_requested: false,
                    quiescent: false,
                })
                .interests
                .insert(demand.clone());
        }
        self.demands.insert(demand, jobs);
        Ok(())
    }
    pub fn release(&mut self, demand: &DemandId) -> Vec<String> {
        let mut cancel = Vec::new();
        if let Some(jobs) = self.demands.remove(demand) {
            for id in jobs {
                if let Some(job) = self.jobs.get_mut(&id) {
                    job.interests.remove(demand);
                    if job.interests.is_empty() {
                        job.cancel_requested = true;
                        cancel.push(id);
                    }
                }
            }
        }
        cancel
    }
    pub fn release_task(&mut self, task: &SessionIdentity) -> Vec<String> {
        let demands: Vec<_> = self
            .demands
            .keys()
            .filter(|d| &d.task == task)
            .cloned()
            .collect();
        demands.iter().flat_map(|d| self.release(d)).collect()
    }
    pub fn mark_quiescent(&mut self, id: &str) -> Result<(), BookError> {
        self.jobs.get_mut(id).ok_or(BookError::Unknown)?.quiescent = true;
        Ok(())
    }
    /// Only drain receipts allow forgotten physical bookkeeping.
    pub fn forget_job(&mut self, id: &str) -> bool {
        if self
            .jobs
            .get(id)
            .is_some_and(|j| j.quiescent && j.interests.is_empty())
        {
            self.jobs.remove(id);
            true
        } else {
            false
        }
    }
    pub fn job(&self, id: &str) -> Option<&JobInterest> {
        self.jobs.get(id)
    }
    pub fn jobs_for(&self, demand: &DemandId) -> Option<&BTreeSet<String>> {
        self.demands.get(demand)
    }
    pub fn counts(&self) -> (usize, usize) {
        (self.demands.len(), self.jobs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn demand(task: &str) -> DemandId {
        DemandId {
            task: SessionIdentity {
                caller_id: "caller".into(),
                session_id: task.into(),
                incarnation: "1".into(),
            },
            id: "prepare".into(),
        }
    }
    #[test]
    fn shared_work_drains_only_after_every_interest_is_released() {
        let mut book = DemandBook::new(2, 1);
        let a = demand("a");
        let b = demand("b");
        book.attach(
            a.clone(),
            BTreeSet::from(["same-prefix-destination-epoch".into()]),
        )
        .unwrap();
        book.attach(
            b.clone(),
            BTreeSet::from(["same-prefix-destination-epoch".into()]),
        )
        .unwrap();
        assert!(book.release(&a).is_empty());
        assert!(
            !book
                .job("same-prefix-destination-epoch")
                .unwrap()
                .cancel_requested
        );
        assert_eq!(
            book.release_task(&b.task),
            vec!["same-prefix-destination-epoch"]
        );
        assert!(!book.forget_job("same-prefix-destination-epoch"));
        assert_eq!(
            book.attach(
                a.clone(),
                BTreeSet::from(["same-prefix-destination-epoch".into()])
            ),
            Err(BookError::Conflict)
        );
        book.mark_quiescent("same-prefix-destination-epoch")
            .unwrap();
        assert!(book.forget_job("same-prefix-destination-epoch"));
        assert_eq!(book.counts(), (0, 0));
    }
    #[test]
    fn fanout_capacity_and_identity_conflicts_are_atomic() {
        let mut book = DemandBook::new(2, 2);
        let a = demand("a");
        let b = demand("b");
        book.attach(a.clone(), BTreeSet::from(["one".into()]))
            .unwrap();
        assert_eq!(
            book.attach(b.clone(), BTreeSet::from(["two".into(), "three".into()])),
            Err(BookError::Capacity)
        );
        assert_eq!(book.counts(), (1, 1));
        assert!(book.jobs_for(&b).is_none());
        assert_eq!(
            book.attach(a.clone(), BTreeSet::from(["two".into()])),
            Err(BookError::Conflict)
        );
        book.attach(a, BTreeSet::from(["one".into()])).unwrap();
        assert_eq!(book.counts(), (1, 1));
    }
}
