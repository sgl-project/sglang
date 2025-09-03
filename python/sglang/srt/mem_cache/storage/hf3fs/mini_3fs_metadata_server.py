import argparse
import atexit
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple

import orjson
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import Hf3fsMetadataInterface

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Data Models ---
class RankMetadata:
    """Holds all metadata for a single rank."""

    def __init__(self, num_pages: int):
        self.lock = threading.Lock()
        self.num_pages = num_pages
        self.free_pages: List[int] = list(range(num_pages))
        self.key_to_index: OrderedDict[str, int] = OrderedDict()
        # Todo: Support multi files for HF3FS

    def exists_keys(self, keys: List[str]) -> List[bool]:
        """Check if keys exist in metadata."""
        with self.lock:
            return [key in self.key_to_index for key in keys]

    def reserve_and_allocate_page_indices(
        self, keys: List[Tuple[str, str]]
    ) -> List[Tuple[bool, int]]:
        """Reserve and allocate page indices for keys."""
        with self.lock:
            results = [None] * len(keys)
            new_keys_to_process = []

            for i, (key, prefix_key) in enumerate(keys):
                if key in self.key_to_index:
                    results[i] = (True, self.key_to_index[key])
                    self.key_to_index.move_to_end(key)
                else:
                    new_keys_to_process.append((i, key, prefix_key))

            # Todo: Implementing data eviction logic after HiCache supports prefix information pass-through
            for i, key, prefix_key in new_keys_to_process:
                if len(self.free_pages) > 0:
                    page_index = self.free_pages.pop()
                else:
                    page_index = self.key_to_index.popitem(last=False)[1]

                results[i] = (False, page_index)

            return results

    def confirm_write(
        self,
        written_keys_to_confirm: List[Tuple[str, int]],
        pages_to_release: List[int],
    ) -> None:
        """Confirm write operations and release pages."""
        with self.lock:
            for key, page_index in written_keys_to_confirm:
                self.key_to_index[key] = page_index
                self.key_to_index.move_to_end(key)

            for page_index in pages_to_release:
                if page_index not in self.free_pages:
                    self.free_pages.append(page_index)

    def delete_keys(self, keys: List[str]) -> int:
        """Delete keys and return count of deleted keys."""
        with self.lock:
            count = 0
            for key in keys:
                if key in self.key_to_index:
                    page_index = self.key_to_index.pop(key)
                    if page_index not in self.free_pages:
                        self.free_pages.append(page_index)
                    count += 1
            return count

    def clear_all(self) -> None:
        """Clear all metadata."""
        with self.lock:
            self.free_pages = list(range(self.num_pages))
            self.key_to_index.clear()

    def get_page_indices(self, keys: List[str]) -> List[Optional[int]]:
        """Get page indices for keys."""
        with self.lock:
            results = []
            for key in keys:
                if key in self.key_to_index:
                    results.append(self.key_to_index[key])
                    self.key_to_index.move_to_end(key)
                else:
                    results.append(None)
            return results


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


# --- Global MetadataServer implementation ---
class Hf3fsMetadataServer:
    """HF3FS Metadata Server that manages metadata for multiple ranks."""

    def __init__(self, persistence_path: Optional[str] = None, save_interval: int = 60):
        self.state = GlobalMetadataState(persistence_path, save_interval)
        self.app = FastAPI(default_response_class=ORJSONResponse)

        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""
        self.app.post("/{rank}/initialize")(self.initialize)
        self.app.post("/{rank}/exists")(self.exists)
        self.app.post("/{rank}/reserve_and_allocate_page_indices")(
            self.reserve_and_allocate_page_indices
        )
        self.app.post("/{rank}/confirm_write")(self.confirm_write)
        self.app.post("/{rank}/delete_keys")(self.delete_keys)
        self.app.post("/{rank}/clear")(self.clear)
        self.app.post("/{rank}/get_page_indices")(self.get_page_indices)

    def get_rank_metadata(self, rank: int) -> RankMetadata:
        """Get rank metadata with proper error handling."""
        if rank not in self.state.ranks:
            raise HTTPException(
                status_code=404,
                detail=f"Rank {rank} not initialized. Please call /{rank}/initialize first.",
            )
        return self.state.ranks[rank]

    async def _read_json(self, request: Request) -> dict:
        """Parse request JSON using orjson if available."""
        body = await request.body()
        return orjson.loads(body)

    def _json_response(self, content: dict):
        """Return ORJSONResponse when available to bypass jsonable_encoder."""
        return ORJSONResponse(content)

    async def initialize(self, rank: int, request: Request):
        """Initialize a rank with specified number of pages."""
        data = await self._read_json(request)
        num_pages = data["num_pages"]
        with self.state.global_lock:
            if rank in self.state.ranks:
                logging.info(
                    f"Rank {rank} already exists. Initialization request ignored."
                )
                if self.state.ranks[rank].num_pages != num_pages:
                    logging.warning(
                        f"Rank {rank} initialized with different num_pages. Existing: {self.state.ranks[rank].num_pages}, New: {num_pages}"
                    )
            else:
                logging.info(f"Initializing new Rank {rank} with {num_pages} pages.")
                self.state.ranks[rank] = RankMetadata(num_pages)
        return Response(status_code=204)

    async def exists(self, rank: int, request: Request):
        """Check if keys exist in metadata."""
        data = await self._read_json(request)
        keys = data["keys"]
        metadata = self.get_rank_metadata(rank)
        results = metadata.exists_keys(keys)
        return self._json_response({"exists": results})

    async def reserve_and_allocate_page_indices(self, rank: int, request: Request):
        """Reserve and allocate page indices for keys."""
        data = await self._read_json(request)
        metadata = self.get_rank_metadata(rank)
        keys = data["keys"]
        results = metadata.reserve_and_allocate_page_indices(keys)
        return self._json_response({"indices": results})

    async def confirm_write(self, rank: int, request: Request):
        """Confirm write operations and release pages."""
        data = await self._read_json(request)
        metadata = self.get_rank_metadata(rank)
        success_written_keys = data.get("written_keys_to_confirm", [])
        released_pages = data.get("pages_to_release", [])

        metadata.confirm_write(success_written_keys, released_pages)

        return Response(status_code=204)

    async def delete_keys(self, rank: int, request: Request):
        """Delete keys from metadata."""
        data = await self._read_json(request)
        metadata = self.get_rank_metadata(rank)
        count = metadata.delete_keys(data["keys"])
        return Response(status_code=204)

    async def clear(self, rank: int):
        """Clear all metadata for a rank."""
        metadata = self.get_rank_metadata(rank)
        metadata.clear_all()
        return Response(status_code=204)

    async def get_page_indices(self, rank: int, request: Request):
        """Get page indices for keys."""
        data = await self._read_json(request)
        metadata = self.get_rank_metadata(rank)
        keys = data["keys"]
        results = metadata.get_page_indices(keys)
        return self._json_response({"indices": results})

    def run(self, host: str = "0.0.0.0", port: int = 18000):
        """Run the metadata server."""
        self.state.load_from_disk()
        if self.state.persistence_path:
            self.state.schedule_save()
            atexit.register(self.state.shutdown)

        import uvicorn

        logging.info(f"Starting metadata server on http://{host}:{port}")
        if self.state.persistence_path:
            logging.info(
                f"Persistence is ENABLED. Saving to '{self.state.persistence_path}' every {self.state.save_interval} seconds."
            )
        else:
            logging.info("Persistence is DISABLED.")

        uvicorn.run(self.app, host=host, port=port)


# --- Client implementation ---
class Hf3fsGlobalMetadataClient(Hf3fsMetadataInterface):
    """Global http metadata client for HF3FS."""

    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=256, pool_maxsize=256
        )
        self._session.mount("http://", adapter)

    def _post(self, endpoint: str, json_data: dict) -> dict:
        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {"Content-Type": "application/json"}
            payload = orjson.dumps(json_data)  # type: ignore[union-attr]
            response = self._session.post(url, data=payload, headers=headers)
            response.raise_for_status()

            if response.status_code == 204 or not response.content:
                return {}
            return orjson.loads(response.content)  # type: ignore[union-attr]
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to POST to {endpoint} after retries: {e}")
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


class Hf3fsLocalMetadataClient(Hf3fsMetadataInterface):
    """Local metadata client that directly operates on single RankMetadata in memory without metadata server."""

    def __init__(self):
        self.rank_metadata = None

    def initialize(self, rank: int, num_pages: int) -> None:
        self.rank_metadata = RankMetadata(num_pages)

    def reserve_and_allocate_page_indices(
        self, rank: int, keys: List[Tuple[str, str]]
    ) -> List[Tuple[bool, int]]:
        """Reserve and allocate page indices for keys."""
        return self.rank_metadata.reserve_and_allocate_page_indices(keys)

    def confirm_write(
        self,
        rank: int,
        written_keys_to_confirm: List[Tuple[str, int]],
        pages_to_release: List[int],
    ) -> None:
        """Confirm write operations."""
        self.rank_metadata.confirm_write(written_keys_to_confirm, pages_to_release)

    def delete_keys(self, rank: int, keys: List[str]) -> None:
        """Delete keys."""
        self.rank_metadata.delete_keys(keys)

    def exists(self, rank: int, keys: List[str]) -> List[bool]:
        """Check if keys exist."""
        return self.rank_metadata.exists_keys(keys)

    def clear(self, rank: int) -> None:
        """Clear all metadata for rank."""
        self.rank_metadata.clear_all()

    def get_page_indices(self, rank: int, keys: List[str]) -> List[Optional[int]]:
        """Get page indices for keys."""
        return self.rank_metadata.get_page_indices(keys)


def run_metadata_server(
    host: str = "0.0.0.0",
    port: int = 18000,
    persistence_path: Optional[str] = None,
    save_interval: int = 60,
):
    """Run the HF3FS metadata server."""
    global server
    server = Hf3fsMetadataServer(
        persistence_path=persistence_path, save_interval=save_interval
    )

    server.run(host=host, port=port)


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

    run_metadata_server(args.host, args.port, args.persistence_path, args.save_interval)
