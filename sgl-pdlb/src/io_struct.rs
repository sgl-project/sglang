use crate::strategy_lb::EngineInfo;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SingleOrBatch<T> {
    Single(T),
    Batch(Vec<T>),
}

pub type InputIds = SingleOrBatch<Vec<i32>>;
pub type InputText = SingleOrBatch<String>;
pub type BootstrapHost = SingleOrBatch<String>;
pub type BootstrapPort = SingleOrBatch<Option<u16>>;
pub type BootstrapRoom = SingleOrBatch<u64>;

#[typetag::serde(tag = "type")]
pub trait Bootstrap {
    fn is_stream(&self) -> bool;
    fn get_batch_size(&self) -> Result<Option<usize>, actix_web::Error>;
    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    );

    fn add_bootstrap_info(&mut self, prefill_info: &EngineInfo) -> Result<(), actix_web::Error> {
        let batch_size = self.get_batch_size()?;
        if let Some(batch_size) = batch_size {
            self.set_bootstrap_info(
                BootstrapHost::Batch(vec![prefill_info.get_hostname(); batch_size]),
                BootstrapPort::Batch(vec![prefill_info.bootstrap_port; batch_size]),
                BootstrapRoom::Batch((0..batch_size).map(|_| rand::random::<u64>()).collect()),
            );
        } else {
            self.set_bootstrap_info(
                BootstrapHost::Single(prefill_info.get_hostname()),
                BootstrapPort::Single(prefill_info.bootstrap_port),
                BootstrapRoom::Single(rand::random::<u64>()),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GenerateReqInput {
    pub text: Option<InputText>,
    pub input_ids: Option<InputIds>,
    #[serde(default)]
    pub stream: bool,
    pub bootstrap_host: Option<BootstrapHost>,
    pub bootstrap_port: Option<BootstrapPort>,
    pub bootstrap_room: Option<BootstrapRoom>,

    #[serde(flatten)]
    pub other: Value,
}

impl GenerateReqInput {
    pub fn get_batch_size(&self) -> Result<Option<usize>, actix_web::Error> {
        if self.text.is_some() && self.input_ids.is_some() {
            return Err(actix_web::error::ErrorBadRequest(
                "Both text and input_ids are present in the request".to_string(),
            ));
        }
        if let Some(InputText::Batch(texts)) = &self.text {
            return Ok(Some(texts.len()));
        }
        if let Some(InputIds::Batch(ids)) = &self.input_ids {
            return Ok(Some(ids.len()));
        }
        Ok(None)
    }
}

#[typetag::serde]
impl Bootstrap for GenerateReqInput {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_batch_size(&self) -> Result<Option<usize>, actix_web::Error> {
        self.get_batch_size()
    }

    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    ) {
        self.bootstrap_host = Some(bootstrap_host);
        self.bootstrap_port = Some(bootstrap_port);
        self.bootstrap_room = Some(bootstrap_room);
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatReqInput {
    #[serde(default)]
    pub stream: bool,
    pub bootstrap_host: Option<BootstrapHost>,
    pub bootstrap_port: Option<BootstrapPort>,
    pub bootstrap_room: Option<BootstrapRoom>,

    #[serde(flatten)]
    pub other: Value,
}

#[typetag::serde]
impl Bootstrap for ChatReqInput {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_batch_size(&self) -> Result<Option<usize>, actix_web::Error> {
        Ok(None)
    }

    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    ) {
        self.bootstrap_host = Some(bootstrap_host);
        self.bootstrap_port = Some(bootstrap_port);
        self.bootstrap_room = Some(bootstrap_room);
    }
}
