use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MultiModalError, MultiModalResult},
    media::{ImageFetchConfig, MediaConnector, MediaSource},
    types::{
        ChatContentPart, ConversationSegment, ImageDetail, Modality, MultiModalData,
        MultiModalUUIDs, PlaceholderHandle, PlaceholderMap, TrackedMedia, DEFAULT_PLACEHOLDERS,
    },
};

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    pub placeholder_tokens: HashMap<Modality, String>,
    pub modality_limits: HashMap<Modality, usize>,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            placeholder_tokens: DEFAULT_PLACEHOLDERS
                .iter()
                .map(|(k, v)| (*k, (*v).to_string()))
                .collect(),
            modality_limits: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub conversation: Vec<ConversationSegment>,
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
    pub placeholders: PlaceholderMap,
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    config: TrackerConfig,
    pending: HashMap<Modality, Vec<PendingTask>>,
    placeholders: PlaceholderMap,
    conversation: Vec<ConversationSegment>,
    uuids: MultiModalUUIDs,
    counts: HashMap<Modality, usize>,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>, config: TrackerConfig) -> Self {
        Self {
            media_connector,
            config,
            pending: HashMap::new(),
            placeholders: PlaceholderMap::new(),
            conversation: Vec::new(),
            uuids: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    pub fn push_part(&mut self, part: ChatContentPart) -> MultiModalResult<()> {
        match part {
            ChatContentPart::Text { text } => {
                if !text.is_empty() {
                    self.conversation.push(ConversationSegment::text(text));
                }
                Ok(())
            }
            ChatContentPart::ImageUrl { url, detail, uuid } => {
                self.enqueue_image(MediaSource::Url(url), detail.unwrap_or_default(), uuid)
            }
            ChatContentPart::ImageData {
                data,
                mime_type: _,
                uuid,
                detail,
            } => self.enqueue_image(
                MediaSource::InlineBytes(data),
                detail.unwrap_or_default(),
                uuid,
            ),
            ChatContentPart::ImageEmbeds { .. } => {
                Err(MultiModalError::UnsupportedContent("image_embeds"))
            }
        }
    }

    pub async fn finalize(mut self) -> MultiModalResult<TrackerOutput> {
        let mut data = MultiModalData::new();
        for (modality, tasks) in self.pending.drain() {
            let mut items = Vec::with_capacity(tasks.len());
            for task in tasks {
                let media = task.await??;
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            conversation: self.conversation,
            data,
            uuids: self.uuids,
            placeholders: self.placeholders,
        })
    }

    fn placeholder_token(&self, modality: Modality) -> &str {
        if let Some(token) = self.config.placeholder_tokens.get(&modality) {
            token.as_str()
        } else {
            DEFAULT_PLACEHOLDERS
                .get(&modality)
                .copied()
                .unwrap_or("<mm>")
        }
    }

    fn next_index(&mut self, modality: Modality) -> MultiModalResult<usize> {
        let count = self.counts.entry(modality).or_insert(0);
        let next = *count;
        let limit = self.config.modality_limits.get(&modality).copied();
        if let Some(limit) = limit {
            if next >= limit {
                return Err(MultiModalError::ModalityLimit { modality, limit });
            }
        }
        *count += 1;
        Ok(next)
    }

    fn record_uuid(&mut self, modality: Modality, uuid: Option<String>) {
        self.uuids.entry(modality).or_default().push(uuid);
    }

    fn add_placeholder(&mut self, token: String, handle: PlaceholderHandle) {
        self.placeholders.entry(token).or_default().push(handle);
    }

    fn enqueue_image(
        &mut self,
        source: MediaSource,
        detail: ImageDetail,
        uuid: Option<String>,
    ) -> MultiModalResult<()> {
        let modality = Modality::Image;
        let idx = self.next_index(modality)?;
        let token = self.placeholder_token(modality).to_string();
        let text_position = self.conversation.len();
        self.conversation
            .push(ConversationSegment::placeholder(token.clone()));
        self.add_placeholder(
            token.clone(),
            PlaceholderHandle {
                modality,
                item_index: idx,
                text_position,
            },
        );
        self.record_uuid(modality, uuid);

        let connector = Arc::clone(&self.media_connector);
        let handle = tokio::spawn(async move {
            let frame = connector
                .fetch_image(source, ImageFetchConfig { detail })
                .await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
        Ok(())
    }
}
