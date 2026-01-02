import asyncio
import dataclasses
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Fields that should always be excluded from request parameters
# because they contain non-JSON-serializable objects (e.g., ImageData, tensors)
ALWAYS_EXCLUDE_FIELDS = {"image_data", "video_data", "audio_data", "input_embeds"}


class RequestMetricsExporter(ABC):
    """Abstract base class for exporting request-level performance metrics to a data destination."""

    def __init__(
        self,
        server_args: ServerArgs,
        obj_skip_names: Optional[set[str]],
        out_skip_names: Optional[set[str]],
    ):
        self.server_args = server_args
        self.obj_skip_names = obj_skip_names or set()
        self.out_skip_names = out_skip_names or set()

    def _format_output_data(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict
    ) -> dict:
        """Format request-level output data containing performance metrics. This method
        should be called prior to writing the data record with `self.write_record()`."""

        request_params = {}
        for field in dataclasses.fields(obj):
            field_name = field.name
            # Skip fields in obj_skip_names or fields that are always excluded (not JSON serializable)
            if (
                field_name not in self.obj_skip_names
                and field_name not in ALWAYS_EXCLUDE_FIELDS
            ):
                value = getattr(obj, field_name)
                # Convert to serializable format
                if value is not None:
                    request_params[field_name] = value

        meta_info = out_dict.get("meta_info", {})
        filtered_out_meta_info = {
            k: v for k, v in meta_info.items() if k not in self.out_skip_names
        }

        request_output_data = {
            "request_parameters": json.dumps(request_params),
            **filtered_out_meta_info,
        }
        return request_output_data

    @abstractmethod
    async def write_record(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict
    ):
        """Write a data record corresponding to a single request, containing performance metric data."""
        pass


class FileRequestMetricsExporter(RequestMetricsExporter):
    """Lightweight `RequestMetricsExporter` implementation that writes records to files on disk.

    Records are written to files in the directory specified by `--export-metrics-to-file-dir`
    server launch flag. File names are of the form `"sglang-request-metrics-{hour_suffix}.log"`.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        obj_skip_names: Optional[set[str]],
        out_skip_names: Optional[set[str]],
    ):
        super().__init__(server_args, obj_skip_names, out_skip_names)
        self.export_dir = getattr(server_args, "export_metrics_to_file_dir")
        os.makedirs(self.export_dir, exist_ok=True)

        # File handler state management
        self._current_file_handler = None
        self._current_hour_suffix = None

    def _ensure_file_handler(self, hour_suffix: str):
        """Ensure the file handler is open for the current hour suffix."""
        if self._current_hour_suffix != hour_suffix:
            # Close previous file handler if it exists
            if self._current_file_handler is not None:
                try:
                    self._current_file_handler.close()
                except Exception as e:
                    logger.warning(f"Failed to close previous file handler: {e}")

            # Open new file handler
            log_filename = f"sglang-request-metrics-{hour_suffix}.log"
            log_filepath = os.path.join(self.export_dir, log_filename)

            try:
                self._current_file_handler = open(log_filepath, "a", encoding="utf-8")
                self._current_hour_suffix = hour_suffix
            except Exception as e:
                logger.error(f"Failed to open log file {log_filepath}: {e}")
                self._current_file_handler = None
                self._current_hour_suffix = None
                raise

    def close(self):
        """Close the current file handler."""
        if self._current_file_handler is not None:
            try:
                self._current_file_handler.close()
            except Exception as e:
                logger.warning(f"Failed to close file handler: {e}")
            finally:
                self._current_file_handler = None
                self._current_hour_suffix = None

    async def write_record(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict
    ):
        # Do not log health check requests, since they don't represent real user requests.
        if isinstance(obj.rid, str) and "HEALTH_CHECK" in obj.rid:
            return

        try:
            # Get the log file path for the current time.
            current_time = datetime.now()
            hour_suffix = current_time.strftime("%Y%m%d_%H")

            # Ensure correct file handler is open for current hour
            self._ensure_file_handler(hour_suffix)

            if self._current_file_handler is None:
                return

            metrics_data = self._format_output_data(obj, out_dict)

            def write_file():
                json.dump(metrics_data, self._current_file_handler)
                self._current_file_handler.write("\n")
                self._current_file_handler.flush()

            await asyncio.to_thread(write_file)
        except Exception as e:
            logger.exception(f"Failed to write perf metrics to file: {e}")


class RequestMetricsExporterManager:
    """Manager class for creating and managing RequestMetricsExporter instances."""

    def __init__(
        self,
        server_args: ServerArgs,
        obj_skip_names: Optional[set[str]] = None,
        out_skip_names: Optional[set[str]] = None,
    ):
        self.server_args = server_args
        self.obj_skip_names = obj_skip_names or set()
        self.out_skip_names = out_skip_names or set()
        self._exporters: List[RequestMetricsExporter] = []
        self._create_exporters()

    def _create_exporters(self) -> None:
        """Create and configure RequestMetricsExporter instances based on server args."""
        # Create standard exporters
        self._exporters.extend(
            create_request_metrics_exporters(
                self.server_args, self.obj_skip_names, self.out_skip_names
            )
        )

        # Import additional RequestMetricsExporter from private fork if available; skip otherwise.
        try:
            from sglang.private.managers.request_metrics_exporter_factory import (
                create_private_request_metrics_exporters,
            )

            self._exporters.extend(
                create_private_request_metrics_exporters(
                    self.server_args, self.obj_skip_names, self.out_skip_names
                )
            )
        except ImportError:
            pass

    def exporter_enabled(self) -> bool:
        """Return true if at least one RequestMetricsExporter is enabled."""
        return len(self._exporters) > 0

    async def write_record(self, obj, out_dict: dict) -> None:
        """Write a record using all configured exporters."""
        for exporter in self._exporters:
            await exporter.write_record(obj, out_dict)


def create_request_metrics_exporters(
    server_args: ServerArgs,
    obj_skip_names: Optional[set[str]] = None,
    out_skip_names: Optional[set[str]] = None,
) -> List[RequestMetricsExporter]:
    """Create and configure `RequestMetricsExporter`s based on server args."""
    metrics_exporters = []

    if server_args.export_metrics_to_file:
        metrics_exporters.append(
            FileRequestMetricsExporter(server_args, obj_skip_names, out_skip_names)
        )

    return metrics_exporters
