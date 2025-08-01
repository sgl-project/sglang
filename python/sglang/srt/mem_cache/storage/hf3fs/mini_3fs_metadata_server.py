import argparse
import atexit
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, HTTPException, Request, status
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import Hf3fsMetadataInterface

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
app = FastAPI()


# --- Data Models ---
class RankMetadata:
    """Holds all metadata for a single rank."""

    def __init__(self, num_pages: int):
        self.lock = threading.RLock()
        self.num_pages = num_pages
        self.free_pages: List[int] = list(range(num_pages))
        self.key_to_index: Dict[str, int] = {}
        # Todo: Support multi files for HF3FS


class GlobalMetadataState:
    """Manages the state for all ranks and persistence."""

    def __init__(self, persistence_path: Optional[str], save_interval: int):
        self.global_lock = threading.RLock()
        self.ranks: Dict[int, RankMetadata] = {}
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.save_interval = save_interval
        self.save_timer: Optional[threading.Timer] = None
        self.is_shutting_down = False

    def load_from_disk(self):
        if not self.persistence_path or not self.persistence_path.exists():
            logging.info("Persistence file not found. Starting with a clean state.")
            return

        logging.info(f"Loading state from {self.persistence_path}")
        try:
            with open(self.persistence_path, "r") as f:
                persisted_data = json.load(f)

            with self.global_lock:
                for rank_id_str, data in persisted_data.items():
                    rank_id = int(rank_id_str)
                    num_pages = data["num_pages"]
                    rank_meta = RankMetadata(num_pages)
                    rank_meta.free_pages = data["free_pages"]
                    rank_meta.key_to_index = dict(data["key_to_index"])
                    self.ranks[rank_id] = rank_meta
                logging.info(
                    f"Successfully loaded metadata for {len(self.ranks)} ranks."
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(
                f"Failed to load or parse persistence file: {e}. Starting fresh.",
                exc_info=True,
            )
            self.ranks.clear()

    def save_to_disk(self):
        if not self.persistence_path:
            return

        logging.info("Persisting metadata to disk...")
        with self.global_lock:
            serializable_state = {}
            for rank_id, rank_meta in self.ranks.items():
                with rank_meta.lock:
                    serializable_state[rank_id] = {
                        "num_pages": rank_meta.num_pages,
                        "free_pages": rank_meta.free_pages,
                        "key_to_index": list(rank_meta.key_to_index.items()),
                    }

        try:
            temp_path = self.persistence_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(serializable_state, f, indent=4)
            temp_path.rename(self.persistence_path)
            logging.info(f"Metadata successfully persisted to {self.persistence_path}")
        except Exception as e:
            logging.error(f"Failed to save metadata to disk: {e}", exc_info=True)

    def schedule_save(self):
        if self.is_shutting_down or not self.persistence_path:
            return
        self.save_to_disk()
        self.save_timer = threading.Timer(self.save_interval, self.schedule_save)
        self.save_timer.start()

    def shutdown(self):
        logging.info("Shutting down metadata server...")
        self.is_shutting_down = True
        if self.save_timer:
            self.save_timer.cancel()
        self.save_to_disk()
        logging.info("Shutdown complete.")


# --- Global State Initialization ---
state = GlobalMetadataState(persistence_path=None, save_interval=60)


def get_rank_metadata(rank: int) -> RankMetadata:
    with state.global_lock:
        if rank not in state.ranks:
            raise HTTPException(
                status_code=404,
                detail=f"Rank {rank} not initialized. Please call /{{rank}}/initialize first.",
            )
        return state.ranks[rank]


# --- API Endpoints ---
@app.post("/{rank}/initialize")
async def initialize(rank: int, request: Request):
    data = await request.json()
    num_pages = data["num_pages"]
    with state.global_lock:
        if rank in state.ranks:
            logging.info(f"Rank {rank} already exists. Initialization request ignored.")
            if state.ranks[rank].num_pages != num_pages:
                logging.warning(
                    f"Rank {rank} initialized with different num_pages. Existing: {state.ranks[rank].num_pages}, New: {num_pages}"
                )
        else:
            logging.info(f"Initializing new Rank {rank} with {num_pages} pages.")
            state.ranks[rank] = RankMetadata(num_pages)
    return {"message": f"Rank {rank} is ready."}


@app.post("/{rank}/exists")
async def exists(rank: int, request: Request):
    data = await request.json()
    keys = data["keys"]
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        results = [key in metadata.key_to_index for key in keys]
        return {"exists": results}


@app.post("/{rank}/reserve_and_allocate_page_indices")
async def reserve_and_allocate_page_indices(rank: int, request: Request):
    data = await request.json()
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        keys = data["keys"]
        results = [None] * len(keys)

        new_keys_to_process = []
        for i, (key, prefix_key) in enumerate(keys):
            if key in metadata.key_to_index:
                results[i] = (True, metadata.key_to_index[key])
            else:
                new_keys_to_process.append((i, key, prefix_key))

        # Todo: Implementing data eviction logic after HiCache supports prefix information pass-through
        for i, key, prefix_key in new_keys_to_process:
            if len(metadata.free_pages) > 0:
                page_idx = metadata.free_pages.pop()
                results[i] = (False, page_idx)
            else:
                results[i] = (False, -1)

        return {"indices": results}


@app.post("/{rank}/confirm_write")
async def confirm_write(rank: int, request: Request):
    data = await request.json()
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        success_written_keys = data.get("written_keys_to_confirm", [])
        released_pages = data.get("pages_to_release", [])

        for key, page_index in success_written_keys:
            metadata.key_to_index[key] = page_index

        for page_index in released_pages:
            if page_index not in metadata.free_pages:
                metadata.free_pages.append(page_index)

    return {
        "message": f"Rank {rank}: Write confirmed for {len(success_written_keys)} keys. {len(released_pages)} pages released."
    }


@app.post("/{rank}/delete_keys")
async def delete_keys(rank: int, request: Request):
    data = await request.json()
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        count = 0
        for key in data["keys"]:
            if key in metadata.key_to_index:
                page_index = metadata.key_to_index.pop(key)
                if page_index not in metadata.free_pages:
                    metadata.free_pages.append(page_index)
                count += 1
    return {"message": f"Rank {rank}: {count} keys deleted."}


@app.post("/{rank}/clear")
async def clear(rank: int):
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        metadata.free_pages = list(range(metadata.num_pages))
        metadata.key_to_index.clear()
    return {"message": f"Rank {rank}: Metadata cleared."}


@app.post("/{rank}/get_page_indices")
async def get_page_indices(rank: int, request: Request):
    data = await request.json()
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        keys = data["keys"]
        results = [metadata.key_to_index.get(key) for key in keys]

        return {"indices": results}


class Hf3fsMetadataClient(Hf3fsMetadataInterface):
    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)

    def _post(self, endpoint: str, json_data: dict) -> dict:
        try:
            response = self._session.post(f"{self.base_url}/{endpoint}", json=json_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to POST to {endpoint} after retries: {e}")
            raise RuntimeError(f"Failed to connect to metadata server: {e}") from e

    def _get(self, endpoint: str) -> dict:
        try:
            response = self._session.get(f"{self.base_url}/{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to GET from {endpoint} after retries: {e}")
            raise RuntimeError(f"Failed to connect to metadata server: {e}") from e

    def initialize(self, rank: int, num_pages: int) -> None:
        self._post(f"{rank}/initialize", {"num_pages": num_pages})

    def reserve_and_allocate_page_indices(
        self, rank: int, keys: List[Tuple[str, str]]
    ) -> List[Tuple[bool, int]]:
        response = self._post(
            f"{rank}/reserve_and_allocate_page_indices", {"keys": keys}
        )
        return [tuple(item) for item in response.get("indices")]

    def confirm_write(
        self,
        rank: int,
        written_keys_to_confirm: List[Tuple[str, int]],
        pages_to_release: List[int],
    ) -> None:
        self._post(
            f"{rank}/confirm_write",
            {
                "written_keys_to_confirm": written_keys_to_confirm,
                "pages_to_release": pages_to_release,
            },
        )

    def delete_keys(self, rank: int, keys: List[str]) -> None:
        self._post(f"{rank}/delete_keys", {"keys": keys})

    def exists(self, rank: int, keys: List[str]) -> List[bool]:
        response = self._post(f"{rank}/exists", {"keys": keys})
        return response.get("exists", [])

    def clear(self, rank: int) -> None:
        self._post(f"{rank}/clear", {})

    def get_page_indices(self, rank: int, keys: List[str]) -> List[Optional[int]]:
        response = self._post(f"{rank}/get_page_indices", {"keys": keys})
        return response.get("indices")


def run(
    host: str = "0.0.0.0",
    port: int = 18000,
    persistence_path: Optional[str] = None,
    save_interval: int = 60,
):
    """Run the HF3FS metadata server."""
    state.persistence_path = Path(persistence_path) if persistence_path else None
    state.save_interval = save_interval

    state.load_from_disk()
    if state.persistence_path:
        state.schedule_save()
        atexit.register(state.shutdown)

    import uvicorn

    logging.info(f"Starting metadata server on http://{host}:{port}")
    if state.persistence_path:
        logging.info(
            f"Persistence is ENABLED. Saving to '{persistence_path}' every {save_interval} seconds."
        )
    else:
        logging.info("Persistence is DISABLED.")

    uvicorn.run(app, host=host, port=port)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF3FS Metadata Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
    )
    parser.add_argument(
        "--port", type=int, default=18000, help="Port to run the server on."
    )
    parser.add_argument(
        "--persistence-path",
        type=str,
        default=None,
        help="Path to the file for persisting metadata. If not provided, persistence is disabled.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=60,
        help="Interval in seconds for periodically saving metadata to disk.",
    )
    args = parser.parse_args()

    run(args.host, args.port, args.persistence_path, args.save_interval)
