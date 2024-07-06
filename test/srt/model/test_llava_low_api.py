import multiprocessing
import time

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.managers.controller.infer_batch import ForwardMode
from sglang.srt.managers.controller.model_runner import InputMetadata, ModelRunner
from sglang.srt.model_config import ModelConfig
from sglang.srt.utils import load_image
from sglang.srt.managers.router.infer_batch import Batch
from sglang.srt.server_args import ServerArgs

def init_batch_data(model, batch_size, input_len):
    req_pool_indices = model.req_to_token_pool.alloc(batch_size)
    seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device="cuda")
    prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    position_ids_offsets = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    out_cache_loc = model.token_to_kv_pool.alloc(batch_size * input_len)
    for i in range(batch_size):
        model.req_to_token_pool.req_to_token[i, :input_len] = out_cache_loc[
            i * input_len : (i + 1) * input_len
        ]

    return (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
    )


def prefill(model, tp_rank, params, print_logits):
    batch_instance=Batch(
      req_pool_indices=params[4],
      seq_lens=params[5],
      prefix_lens=params[6],
      position_ids_offsets=params[7],
      out_cache_loc=params[8],
      input_ids=params[0],
      pixel_values=params[1],
      image_sizes=params[2],
      image_offsets=params[3],
      top_logprobs_nums=None,
      return_logprob=False,
      reqs=None,
      req_to_token_pool=None,
      token_to_kv_pool=None,
      tree_cache=None

    )

    logits, _ = model.forward_extend_multi_modal(
        batch_instance
    )
    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    if print_logits and tp_rank == 0:
        print("prefill logits", logits, logits.shape)

    return predict_ids


def decode(step, model, tp_rank, batch_size, predict_ids, params, print_logits):
    (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
    ) = params

    (
        out_cache_loc,
        out_cache_cont_start,
        out_cache_cont_end,
    ) = model.token_to_kv_pool.alloc_contiguous(batch_size)
    model.req_to_token_pool.req_to_token[req_pool_indices, seq_lens] = out_cache_loc
    seq_lens.add_(1)
    batch_instance2=Batch(
      req_pool_indices=req_pool_indices,
      seq_lens=seq_lens,
      prefix_lens=None,
      position_ids_offsets=position_ids_offsets,
      out_cache_loc=None,
      input_ids=torch.from_numpy(predict_ids).cuda().reshape(-1),
      out_cache_cont_start=out_cache_cont_start,
      out_cache_cont_end=out_cache_cont_end,
      top_logprobs_nums=None,
      return_logprob=False,
      reqs=None,
      req_to_token_pool=None,
      token_to_kv_pool=None,
      tree_cache=None
    )
    logits, _ = model.forward_decode(
      batch_instance2
    )
    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()
    if print_logits and tp_rank == 0:
        print("decode", step, logits)
    return predict_ids


def test_generate_worker(
    model_path,
    tp_rank,
    tp_size,
):
    model_config = ModelConfig(path=model_path)
    model = ModelRunner(server_args=ServerArgs(model_path=model_path),model_config=model_config, mem_fraction_static=0.8, tp_rank=tp_rank, tp_size=tp_size,gpu_id=0,nccl_port=28888,)
    # print(model.model)

    # Prepare data
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this picture ASSISTANT:"
    image_path = "/content/drive/MyDrive/sglang2/test/srt/example_image.png"
    image,_= load_image(image_path)

    processor = get_processor("llava-hf/llava-1.5-7b-hf")
    input_ids = processor.tokenizer.encode(prompt)
    pixel_values = processor.image_processor(image)["pixel_values"]
    input_ids, offset = model.model.pad_input_ids(
        input_ids,
        [
            0,
        ],
    )

    params = init_batch_data(model, 1, len(input_ids))

    # inference
    output_ids = []
    prefill_params = (
        torch.tensor(np.array(input_ids)).cuda(),
        np.array(pixel_values),
        [None],
        [offset],
        *params,
    )
    predict_ids = prefill(model, tp_rank=0, params=prefill_params, print_logits=False)
    output_ids.append(predict_ids[0][0])
    for i in range(16):
        predict_ids = decode(
            i,
            model,
            tp_rank=0,
            batch_size=1,
            predict_ids=predict_ids,
            params=params,
            print_logits=False,
        )
        output_ids.append(predict_ids[0][0])

    # detokenization
    output = processor.tokenizer.batch_decode(
        [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    assert (
        output
        == "The image features a man standing on the back of a yellow taxi cab, holding"
    )


def test_generate(model_path, tp_size):
    workers = []
    for tp_rank in range(tp_size):
        proc = multiprocessing.Process(
            target=test_generate_worker,
            args=(
                model_path,
                tp_rank,
                tp_size,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()


if __name__ == "__main__":
    test_generate("liuhaotian/llava-v1.5-7b", 1)
