use std::{
    collections::{BTreeMap, HashMap},
    sync::RwLock,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use super::{
    conversation_items::{
        make_item_id, ConversationItem, ConversationItemId, ConversationItemStorage, ListParams,
        Result, SortOrder,
    },
    conversations::ConversationId,
};

#[derive(Default)]
pub struct MemoryConversationItemStorage {
    items: RwLock<HashMap<ConversationItemId, ConversationItem>>, // item_id -> item
    #[allow(clippy::type_complexity)]
    links: RwLock<HashMap<ConversationId, BTreeMap<(i64, String), ConversationItemId>>>,
    // Per-conversation reverse index for fast after cursor lookup: item_id_str -> (ts, item_id_str)
    #[allow(clippy::type_complexity)]
    rev_index: RwLock<HashMap<ConversationId, HashMap<String, (i64, String)>>>,
}

impl MemoryConversationItemStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ConversationItemStorage for MemoryConversationItemStorage {
    async fn create_item(
        &self,
        new_item: super::conversation_items::NewConversationItem,
    ) -> Result<ConversationItem> {
        let id = new_item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&new_item.item_type));
        let created_at = Utc::now();
        let item = ConversationItem {
            id: id.clone(),
            response_id: new_item.response_id,
            item_type: new_item.item_type,
            role: new_item.role,
            content: new_item.content,
            status: new_item.status,
            created_at,
        };
        let mut items = self.items.write().unwrap();
        items.insert(id.clone(), item.clone());
        Ok(item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> Result<()> {
        {
            let mut links = self.links.write().unwrap();
            let entry = links.entry(conversation_id.clone()).or_default();
            entry.insert((added_at.timestamp(), item_id.0.clone()), item_id.clone());
        }
        {
            let mut rev = self.rev_index.write().unwrap();
            let entry = rev.entry(conversation_id.clone()).or_default();
            entry.insert(item_id.0.clone(), (added_at.timestamp(), item_id.0.clone()));
        }
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> Result<Vec<ConversationItem>> {
        let links_guard = self.links.read().unwrap();
        let map = match links_guard.get(conversation_id) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let mut results: Vec<ConversationItem> = Vec::new();
        let after_key: Option<(i64, String)> = if let Some(after_id) = &params.after {
            // O(1) lookup via reverse index for this conversation
            if let Some(conv_idx) = self.rev_index.read().unwrap().get(conversation_id) {
                conv_idx.get(after_id).cloned()
            } else {
                None
            }
        } else {
            None
        };

        let take = params.limit;
        let items_guard = self.items.read().unwrap();

        use std::ops::Bound::{Excluded, Unbounded};

        // Helper to push item if it exists and stop when reaching the limit
        let mut push_item = |key: &ConversationItemId| -> bool {
            if let Some(it) = items_guard.get(key) {
                results.push(it.clone());
                if results.len() == take {
                    return true;
                }
            }
            false
        };

        match (params.order, after_key) {
            (SortOrder::Desc, Some(k)) => {
                for ((_ts, _id), item_key) in map.range(..k).rev() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Desc, None) => {
                for ((_ts, _id), item_key) in map.iter().rev() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Asc, Some(k)) => {
                for ((_ts, _id), item_key) in map.range((Excluded(k), Unbounded)) {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Asc, None) => {
                for ((_ts, _id), item_key) in map.iter() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    async fn get_item(&self, item_id: &ConversationItemId) -> Result<Option<ConversationItem>> {
        let items = self.items.read().unwrap();
        Ok(items.get(item_id).cloned())
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<bool> {
        let rev = self.rev_index.read().unwrap();
        if let Some(conv_idx) = rev.get(conversation_id) {
            Ok(conv_idx.contains_key(&item_id.0))
        } else {
            Ok(false)
        }
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<()> {
        // Get the key from rev_index and remove the entry at the same time
        let key_to_remove = {
            let mut rev = self.rev_index.write().unwrap();
            if let Some(conv_idx) = rev.get_mut(conversation_id) {
                conv_idx.remove(&item_id.0)
            } else {
                None
            }
        };

        // If the item was in rev_index, remove it from links as well
        if let Some(key) = key_to_remove {
            let mut links = self.links.write().unwrap();
            if let Some(conv_links) = links.get_mut(conversation_id) {
                conv_links.remove(&key);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;

    fn make_item(
        item_type: &str,
        role: Option<&str>,
        content: serde_json::Value,
    ) -> super::super::conversation_items::NewConversationItem {
        super::super::conversation_items::NewConversationItem {
            id: None,
            response_id: None,
            item_type: item_type.to_string(),
            role: role.map(|r| r.to_string()),
            content,
            status: Some("completed".to_string()),
        }
    }

    #[tokio::test]
    async fn test_list_ordering_and_cursors() {
        let store = MemoryConversationItemStorage::new();
        let conv: ConversationId = "conv_test".into();

        // Create 3 items and link them at controlled timestamps
        let i1 = store
            .create_item(make_item("message", Some("user"), serde_json::json!([])))
            .await
            .unwrap();
        let i2 = store
            .create_item(make_item(
                "message",
                Some("assistant"),
                serde_json::json!([]),
            ))
            .await
            .unwrap();
        let i3 = store
            .create_item(make_item("reasoning", None, serde_json::json!([])))
            .await
            .unwrap();

        let t1 = Utc.timestamp_opt(1_700_000_001, 0).single().unwrap();
        let t2 = Utc.timestamp_opt(1_700_000_002, 0).single().unwrap();
        let t3 = Utc.timestamp_opt(1_700_000_003, 0).single().unwrap();

        store.link_item(&conv, &i1.id, t1).await.unwrap();
        store.link_item(&conv, &i2.id, t2).await.unwrap();
        store.link_item(&conv, &i3.id, t3).await.unwrap();

        // Desc order, no cursor
        let desc = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Desc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(desc.len() >= 2);
        assert_eq!(desc[0].id, i3.id);
        assert_eq!(desc[1].id, i2.id);

        // Desc with cursor = i2 -> expect i1 next
        let desc_after = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Desc,
                    after: Some(i2.id.0.clone()),
                },
            )
            .await
            .unwrap();
        assert!(!desc_after.is_empty());
        assert_eq!(desc_after[0].id, i1.id);

        // Asc order, no cursor
        let asc = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(asc.len() >= 2);
        assert_eq!(asc[0].id, i1.id);
        assert_eq!(asc[1].id, i2.id);

        // Asc with cursor = i2 -> expect i3 next
        let asc_after = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Asc,
                    after: Some(i2.id.0.clone()),
                },
            )
            .await
            .unwrap();
        assert!(!asc_after.is_empty());
        assert_eq!(asc_after[0].id, i3.id);
    }
}
