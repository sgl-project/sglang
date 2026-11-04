//! Immutable selector binding. `latest` is resolved once; a target never follows
//! future branch updates. The index owns token material, so a frozen target may
//! remain logically known after material eviction while execution reports gaps.
use crate::{
    models::{PrefixTarget, SessionIdentity},
    protocol::MAX_INTEGER,
    session_index::{
        BranchRevision, ContentKey, ContentNode, RequestIdentity, SessionIndex, SnapshotId,
    },
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Selector {
    Session(SessionIdentity),
    Branch {
        session: SessionIdentity,
        branch_id: String,
        revision: Option<u64>,
    },
    Request {
        session: SessionIdentity,
        request: RequestIdentity,
    },
    /// Adapter registers the content DAG (with optional tokens) before resolving.
    /// Opaque digest construction and event/storage key translation stay outside.
    Prefix {
        content: ContentKey,
        start_token: u64,
        end_token: u64,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MissingScope {
    Session(SessionIdentity),
    Branch {
        session: SessionIdentity,
        branch_id: String,
        revision: Option<u64>,
    },
    Request {
        session: SessionIdentity,
        request: RequestIdentity,
    },
    Content(ContentKey),
    EmptySession(SessionIdentity),
    InvalidRange {
        content: ContentKey,
        start_token: u64,
        end_token: u64,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaterializationGap {
    Tokens {
        content: ContentKey,
        start_token: u64,
        end_token: u64,
    },
    /// P1 execution requires [0,end); arbitrary middle intervals are unsupported.
    FullPrefixRequired {
        start_token: u64,
        end_token: u64,
    },
    ProtocolTokenLimit {
        end_token: u64,
        max_tokens: u64,
    },
}

#[derive(Clone, Debug)]
pub struct ContentScope {
    content: Arc<ContentNode>,
    start_token: u64,
    end_token: u64,
}
impl ContentScope {
    pub fn content(&self) -> &Arc<ContentNode> {
        &self.content
    }
    pub fn key(&self) -> &ContentKey {
        self.content.key()
    }
    pub fn start_token(&self) -> u64 {
        self.start_token
    }
    pub fn end_token(&self) -> u64 {
        self.end_token
    }
    pub(crate) fn new(content: Arc<ContentNode>, start_token: u64, end_token: u64) -> Self {
        Self {
            content,
            start_token,
            end_token,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ResolvedScope {
    /// None for explicit prefix selection; branch identity is never guessed.
    snapshot: Option<Arc<BranchRevision>>,
    scope: ContentScope,
}
impl ResolvedScope {
    pub fn snapshot(&self) -> Option<&Arc<BranchRevision>> {
        self.snapshot.as_ref()
    }
    pub fn scope(&self) -> &ContentScope {
        &self.scope
    }
}
#[derive(Clone, Debug)]
pub struct FrozenTarget {
    snapshot_revision: Option<u64>,
    known_scope: Vec<ResolvedScope>,
    missing_scope: Vec<MissingScope>,
    materialization_required: Vec<MaterializationGap>,
}
impl FrozenTarget {
    pub fn snapshot_revision(&self) -> Option<u64> {
        self.snapshot_revision
    }
    pub fn known_scope(&self) -> &[ResolvedScope] {
        &self.known_scope
    }
    pub fn missing_scope(&self) -> &[MissingScope] {
        &self.missing_scope
    }
    /// Gaps as of binding time. Call `materialize` again before an Engine submission.
    pub fn materialization_required(&self) -> &[MaterializationGap] {
        &self.materialization_required
    }
    pub fn is_complete(&self) -> bool {
        self.missing_scope.is_empty() && self.materialization_required.is_empty()
    }
}

impl SessionIndex {
    pub fn resolve(&self, selector: Selector) -> FrozenTarget {
        let mut result = FrozenTarget {
            snapshot_revision: None,
            known_scope: Vec::new(),
            missing_scope: Vec::new(),
            materialization_required: Vec::new(),
        };
        match selector {
            Selector::Session(id) => match self.branch_ids(&id) {
                None => result.missing_scope.push(MissingScope::Session(id)),
                Some(ids) => {
                    result.snapshot_revision = self.session_revision(&id);
                    if ids.is_empty() {
                        result.missing_scope.push(MissingScope::EmptySession(id));
                    }
                    for branch in ids {
                        if let Some(snapshot) =
                            self.branch(&branch.session, &branch.branch_id, Some(branch.revision))
                        {
                            result.push_snapshot(snapshot);
                        }
                    }
                }
            },
            Selector::Branch {
                session,
                branch_id,
                revision,
            } => {
                result.snapshot_revision = self.session_revision(&session);
                if let Some(snapshot) = self.branch(&session, &branch_id, revision) {
                    result.push_snapshot(snapshot);
                } else {
                    result.missing_scope.push(MissingScope::Branch {
                        session,
                        branch_id,
                        revision,
                    });
                }
            }
            Selector::Request { session, request } => {
                result.snapshot_revision = self.session_revision(&session);
                if let Some(snapshot) = self.request(&session, &request) {
                    result.push_snapshot(snapshot);
                } else {
                    result
                        .missing_scope
                        .push(MissingScope::Request { session, request });
                }
            }
            Selector::Prefix {
                content,
                start_token,
                end_token,
            } => {
                if let Some(node) = self.content(&content) {
                    let page = content.compatibility().page_size;
                    if start_token >= end_token
                        || end_token > MAX_INTEGER
                        || end_token > node.end_token()
                        || start_token % page != 0
                        || end_token % page != 0
                    {
                        result.missing_scope.push(MissingScope::InvalidRange {
                            content,
                            start_token,
                            end_token,
                        });
                    } else {
                        result.known_scope.push(ResolvedScope {
                            snapshot: None,
                            scope: ContentScope::new(node, start_token, end_token),
                        });
                    }
                } else {
                    result.missing_scope.push(MissingScope::Content(content));
                }
            }
        }
        for scope in &result.known_scope {
            result
                .materialization_required
                .extend(self.materialization_gaps(&scope.scope));
        }
        result
    }
    pub fn materialization_gaps(&self, scope: &ContentScope) -> Vec<MaterializationGap> {
        let mut gaps = Vec::new();
        if scope.start_token != 0 {
            gaps.push(MaterializationGap::FullPrefixRequired {
                start_token: scope.start_token,
                end_token: scope.end_token,
            });
        }
        if scope.end_token > 65_536 {
            gaps.push(MaterializationGap::ProtocolTokenLimit {
                end_token: scope.end_token,
                max_tokens: 65_536,
            });
        }
        let mut node = Some(scope.content.as_ref());
        while let Some(n) = node {
            if n.start_token() < scope.end_token && self.tokens(n.key()).is_none() {
                gaps.push(MaterializationGap::Tokens {
                    content: n.key().clone(),
                    start_token: n.start_token(),
                    end_token: n.end_token().min(scope.end_token),
                });
            }
            node = n.parent().map(Arc::as_ref);
        }
        gaps.sort_by_key(|g| match g {
            MaterializationGap::Tokens { start_token, .. }
            | MaterializationGap::FullPrefixRequired { start_token, .. } => *start_token,
            MaterializationGap::ProtocolTokenLimit { .. } => 0,
        });
        gaps
    }
    /// A fresh allocation only at the Engine protocol boundary. Stored branch
    /// history and frozen operation targets never duplicate the full token prefix.
    pub fn materialize(
        &self,
        scope: &ContentScope,
    ) -> Result<PrefixTarget, Vec<MaterializationGap>> {
        let gaps = self.materialization_gaps(scope);
        if !gaps.is_empty() {
            return Err(gaps);
        }
        let mut chain = Vec::new();
        let mut node = Some(scope.content.as_ref());
        while let Some(n) = node {
            chain.push(n);
            node = n.parent().map(Arc::as_ref);
        }
        let mut tokens = Vec::new();
        for n in chain.into_iter().rev() {
            if n.start_token() >= scope.end_token {
                break;
            }
            let suffix = self
                .tokens(n.key())
                .expect("materialization checked while borrowed");
            let count = (n.end_token().min(scope.end_token) - n.start_token()) as usize;
            tokens.extend_from_slice(&suffix[..count]);
        }
        Ok(PrefixTarget {
            kind: "prefix".into(),
            compatibility: scope.key().compatibility().clone(),
            start_token: scope.start_token,
            end_token: scope.end_token,
            token_ids: Some(tokens),
            blocks: Vec::new(),
        })
    }
}
impl FrozenTarget {
    fn push_snapshot(&mut self, snapshot: Arc<BranchRevision>) {
        let scope = ContentScope::new(
            snapshot.content().clone(),
            0,
            snapshot.content().end_token(),
        );
        self.known_scope.push(ResolvedScope {
            snapshot: Some(snapshot),
            scope,
        });
    }
    pub fn branch_set(&self) -> Vec<SnapshotId> {
        self.known_scope
            .iter()
            .filter_map(|s| s.snapshot.as_ref().map(|b| b.id().clone()))
            .collect()
    }
}
