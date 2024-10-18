import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

from sglang.srt.router.router import BaseRouter, get_router_class
from sglang.srt.router.worker import Worker


def configure_logger(log_level, prefix: str = ""):
    # add level to the format
    format = f"[%(asctime)s{prefix}] %(levelname)s: %(message)s"

    logging.basicConfig(
        level=log_level,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


@dataclass
class WorkerInfo:
    server_url: str
