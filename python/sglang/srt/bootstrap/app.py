#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: app.py
@time: 2025/03/24
@contact: ybyang7@iflytek.com
@site:
@software: PyCharm

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
from pydantic import BaseModel


import threading
import logging
from typing import Optional, Dict
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse



#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# Define request and response data models
class OpenSessionReqInput(BaseModel):
    session_id: Optional[str] = None
    # Additional fields...

class HandshakeRequest(BaseModel):
    room_id: int
    session_id: str
    engine_rank: int
    ib_device: str
    ip_addr: dict

class PrefillReadyRequest(BaseModel):
    room_id: int
    ready: bool = True

# Mapping from room_id to rdma_port
room_to_port_mapping = {}
# Set of room_ids that are ready for prefill
prefill_ready_rooms = set()

def _create_error_response(e):
    return JSONResponse(
        status_code=500,
        content={"error": str(e)},
    )

# Initialize FastAPI application
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/handshake", methods=["POST"])
async def handshake(request: HandshakeRequest):
    """
    Handle handshake request from the receiver
    The receiver establishes a RDMA verbs server and sends room_id and rdma_port information
    """
    try:
        room_id = request.room_id
        global room_to_port_mapping
        logging.info(f"Handshake successful for room_id: {room_id}")
        if room_id not in room_to_port_mapping:
            room_to_port_mapping[room_id] = {request.engine_rank: request.ip_addr}
        else:
            room_to_port_mapping[room_id].update({request.engine_rank:request.ip_addr})
        return {
            "status": "success",
            "message": f"Handshake completed for room {room_id}",
            "room_id": room_id,
        }
    except Exception as e:
        logging.error(f"Handshake failed: {str(e)}")
        return _create_error_response(e)

@app.api_route("/get_room_info/{room_id}", methods=["GET"])
async def get_room_info(room_id: int):
    """
    Query rdma_port information for a specific room_id
    Sender can use this endpoint to get the receiver's rdma_port
    """
    try:
        if room_id not in room_to_port_mapping:
            return JSONResponse(
                status_code=404,
                content={"error": f"Room {room_id} not found"}
            )

        return room_to_port_mapping.get(room_id, {})
    except Exception as e:
        return _create_error_response(e)




class BootstrapServerStarter:
    def __init__(self, app: FastAPI, host: str, port: int, shared_data: dict):
        self.app = app
        self.host = host
        self.port = port
        # Store shared data accessible from both threads
        self.app.shared_data = shared_data
        self.server = None
        self.thread = None

    def start(self):
        """Start the server in a separate thread"""
        self.thread = threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def _run(self):
        """Internal method to run the server"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=logging.INFO,
            timeout_keep_alive=5,
            loop="uvloop"
        )


def start_bootstrap_server(bootstrap_host: str, bootstrap_port: int, server_args: Optional[dict] = None):
    """
    Start the bootstrap server in a separate thread with shared data

    Args:
        bootstrap_host: Host address for the server
        bootstrap_port: Port number for the server
        server_args: Optional additional server configuration arguments

    Returns:
        tuple: (UvicornServer instance, shared data dictionary)
    """
    server = BootstrapServerStarter(app, bootstrap_host, bootstrap_port, {})
    server.start()

    return server
