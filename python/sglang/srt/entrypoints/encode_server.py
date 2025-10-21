import uvicorn
import pickle

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from transformers import AutoImageProcessor
from transformers.image_utils import load_images
from typing import Optional

from sglang.srt.server_args import PortArgs, ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader import get_model
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.distributed.parallel_state import initialize_model_parallel, init_distributed_environment
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.schedule_batch import MultimodalDataItem, Modality

class ImageEncoder:
    def __init__(self, server_args:ServerArgs):
        set_global_server_args_for_scheduler(server_args)
        
        self.image_processor = AutoImageProcessor.from_pretrained(
            server_args.model_path, trust_remote_code=server_args.trust_remote_code
        )
        
        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        
        port_args = PortArgs.init_new(server_args)
        if server_args.dist_init_addr:
            dist_init_method = f"tcp://{server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{port_args.nccl_port}"
        
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=dist_init_method)
        initialize_model_parallel()
        initialize_dp_attention(server_args, self.model_config)
        
        self.model = get_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=DeviceConfig(),
            )
        
    def encoding(self,mm_items):
        images = load_images(mm_items)
        images_input = self.image_processor(images=images)
        mm_item = MultimodalDataItem.from_dict({
            'modality':Modality.IMAGE,
            'feature':images_input['pixel_values'],
        })
        mm_item.set('image_grid_thw', images_input['image_grid_thw'])
        mm_embeddings = self.model.get_image_feature([mm_item])
        return mm_embeddings

app = FastAPI()
encoder: Optional[ImageEncoder] = None
        
def launch_server(server_args:ServerArgs):
    global encoder
    encoder = ImageEncoder(server_args)
    uvicorn.run(app, host=server_args.host, port=server_args.port)

@app.post("/encode")
async def handle_encode_request(request_data: dict):
    mm_embeddings = encoder.encoding(request_data['mm_items'])
    return ORJSONResponse(content={'mm_embeddings':mm_embeddings.tolist()})