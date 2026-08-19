//! Engine-free text request preparation workers for the standalone renderer.

use std::sync::Arc;

use tokio::sync::oneshot;

use crate::message::request::GenerateRequest;
use crate::runtime::Runnable;
use crate::tokenizer_manager::to_scheduler::{
    Limits, check_total_tokens, validate_generate_request,
};
use crate::tokenizer_manager::tokenizer::{TextTokenizer, tokenize_generate_request};
use crate::utils::error::Error;

pub(crate) struct RenderJob {
    pub(crate) request: GenerateRequest,
    pub(crate) reply: oneshot::Sender<Result<GenerateRequest, Error>>,
}

pub(crate) struct RenderWorker {
    jobs: flume::Receiver<RenderJob>,
    tokenizer: Arc<dyn TextTokenizer>,
    auto_specials: Vec<i32>,
    limits: Limits,
}

impl RenderWorker {
    pub(crate) fn new(
        jobs: flume::Receiver<RenderJob>,
        tokenizer: Arc<dyn TextTokenizer>,
        limits: Limits,
    ) -> Self {
        let auto_specials = tokenizer.auto_specials();
        Self {
            jobs,
            tokenizer,
            auto_specials,
            limits,
        }
    }

    fn prepare(&self, mut request: GenerateRequest) -> Result<GenerateRequest, Error> {
        validate_generate_request(&request.rid, &request, &self.limits)?;
        if request.has_multimodal() {
            return Err(Error::Validation(
                "multimodal inputs are not supported by the standalone renderer".into(),
            ));
        }
        request
            .sampling_params
            .normalize(self.limits.skip_tokenizer_init, self.limits.vocab_size)?;
        if !request.already_tokenized() {
            tokenize_generate_request(&mut request, self.tokenizer.as_ref(), &self.auto_specials)?;
        }
        check_total_tokens(&mut request, &self.limits)?;
        Ok(request)
    }
}

impl Runnable for RenderWorker {
    fn run(self) {
        while let Ok(job) = self.jobs.recv() {
            let result = self.prepare(job.request);
            // The HTTP request may have been cancelled while preprocessing.
            let _ = job.reply.send(result);
        }
    }
}
