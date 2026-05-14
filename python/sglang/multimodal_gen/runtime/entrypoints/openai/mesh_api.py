import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    MeshGenerationsRequest,
    MeshListResponse,
    MeshResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import MESH_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/meshes", tags=["meshes"])


def _normalize_format(fmt: Optional[str]) -> str:
    fmt = (fmt or "glb").lower()
    return fmt if fmt in ("glb", "obj") else "glb"


def _build_sampling_params_from_request(
    request_id: str, req: MeshGenerationsRequest, image_path: Optional[str] = None
) -> SamplingParams:
    ext = _normalize_format(req.output_format)

    server_args = get_global_server_args()
    sampling_kwargs: Dict[str, Any] = {
        "request_id": request_id,
        "prompt": req.prompt,
        "num_frames": 1,
        "image_path": [image_path] if image_path else None,
        "save_output": True,
        "output_file_name": f"{request_id}.{ext}",
        "seed": req.seed,
        "generator_device": req.generator_device,
    }
    if req.num_inference_steps is not None:
        sampling_kwargs["num_inference_steps"] = req.num_inference_steps
    if req.guidance_scale is not None:
        sampling_kwargs["guidance_scale"] = req.guidance_scale
    if req.negative_prompt is not None:
        sampling_kwargs["negative_prompt"] = req.negative_prompt

    return SamplingParams.from_user_sampling_params_args(
        model_path=server_args.model_path,
        server_args=server_args,
        **sampling_kwargs,
    )


def _mesh_job_from_sampling(
    request_id: str, req: MeshGenerationsRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "mesh",
        "model": req.model or "",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "format": _normalize_format(req.output_format),
        "file_path": os.path.abspath(sampling.output_file_path()),
    }


async def _dispatch_job_async(job_id: str, batch: Req) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]

        file_size = None
        if os.path.exists(save_file_path):
            file_size = os.path.getsize(save_file_path)

        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

        update_fields: Dict[str, Any] = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
            "url": cloud_url,
            "file_path": save_file_path if not cloud_url else None,
            "file_size_bytes": file_size,
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await MESH_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.error(f"{e}")
        await MESH_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )


@router.post("", response_model=MeshResponse)
async def create_mesh(
    request: Request,
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    url: Optional[List[str]] = Form(None),
    url_array: Optional[List[str]] = Form(None, alias="url[]"),
    prompt: Optional[str] = Form("generate 3d mesh"),
    model: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    generator_device: Optional[str] = Form("cuda"),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    output_format: Optional[str] = Form("glb"),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()
    server_args = get_global_server_args()

    input_path = None

    if "multipart/form-data" in content_type:
        images = image or image_array
        urls = url or url_array
        image_list = merge_image_input_list(images, urls)

        if not image_list:
            raise HTTPException(
                status_code=422,
                detail="Field 'image' or 'url' is required for mesh generation",
            )

        uploads_dir = os.path.join("outputs", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        img = image_list[0]
        filename = img.filename if hasattr(img, "filename") else "input_image"
        try:
            input_path = await save_image_to_path(
                img, os.path.join(uploads_dir, f"{request_id}_{filename}")
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to process image source: {str(e)}"
            )

        req = MeshGenerationsRequest(
            prompt=prompt or "generate 3d mesh",
            model=model,
            seed=seed,
            generator_device=generator_device,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            output_format=output_format,
            **(
                {"guidance_scale": guidance_scale} if guidance_scale is not None else {}
            ),
        )
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            payload: Dict[str, Any] = dict(body or {})

            if payload.get("input_image"):
                img_src = payload.pop("input_image")
                uploads_dir = os.path.join("outputs", "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                input_path = await save_image_to_path(
                    img_src,
                    os.path.join(uploads_dir, f"{request_id}_input_image"),
                )

            req = MeshGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    if not input_path:
        raise HTTPException(
            status_code=422,
            detail="An input image is required for mesh generation",
        )

    sampling_params = _build_sampling_params_from_request(request_id, req, input_path)
    job = _mesh_job_from_sampling(request_id, req, sampling_params)
    await MESH_STORE.upsert(request_id, job)

    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
    )

    asyncio.create_task(_dispatch_job_async(request_id, batch))
    return MeshResponse(**job)


@router.get("", response_model=MeshListResponse)
async def list_meshes(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await MESH_STORE.list_values()

    reverse = order != "asc"
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=reverse)

    if after is not None:
        try:
            idx = next(i for i, j in enumerate(jobs) if j["id"] == after)
            jobs = jobs[idx + 1 :]
        except StopIteration:
            jobs = []

    if limit is not None:
        jobs = jobs[:limit]
    items = [MeshResponse(**j) for j in jobs]
    return MeshListResponse(data=items)


@router.get("/{mesh_id}", response_model=MeshResponse)
async def retrieve_mesh(mesh_id: str = Path(...)):
    job = await MESH_STORE.get(mesh_id)
    if not job:
        raise HTTPException(status_code=404, detail="Mesh not found")
    return MeshResponse(**job)


@router.delete("/{mesh_id}", response_model=MeshResponse)
async def delete_mesh(mesh_id: str = Path(...)):
    job = await MESH_STORE.pop(mesh_id)
    if not job:
        raise HTTPException(status_code=404, detail="Mesh not found")
    job["status"] = "deleted"
    return MeshResponse(**job)


@router.get("/{mesh_id}/content")
async def download_mesh_content(
    mesh_id: str = Path(...), variant: Optional[str] = Query(None)
):
    job = await MESH_STORE.get(mesh_id)
    if not job:
        raise HTTPException(status_code=404, detail="Mesh not found")

    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=f"Mesh has been uploaded to cloud storage. Please use the cloud URL: {job.get('url')}",
        )

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    ext = os.path.splitext(file_path)[1].lower()
    media_type = {
        ".glb": "model/gltf-binary",
        ".obj": "text/plain",
    }.get(ext, "application/octet-stream")

    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
