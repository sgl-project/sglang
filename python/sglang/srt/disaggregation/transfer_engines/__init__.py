#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: __init__.py
@time: 2025/04/07
@contact: ybyang7@iflytek.com
@site:
@software: PyCharm

# Code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  God Bless   ┣┓
                ┃  No Bugs!    ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

# Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
# Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
# Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
# Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
# Vestibulum commodo. Ut rhoncus gravida arcu.
import importlib
import logging

logger = logging.getLogger(__name__)

def load_transfer_engine_classes(impl: str):
    """
    Load corresponding KVManager, KVSender, KVReceiver implementation classes based on the implementation name.
    :param impl: e.g., 'rdma', 'fake', 'mlark'
    :return: (KVManagerClass, KVSenderClass, KVReceiverClass, KVArgsClass, KVBootstrapServerClass)
    """
    module_name = f"sglang.srt.disaggregation.transfer_engines.{impl}_impl"
    logger.info(f"Loading {impl} kv transfer engine")
    module = importlib.import_module(module_name)

    KVManagerClass = getattr(module, "KVManager")
    KVSenderClass = getattr(module, "KVSender")
    KVReceiverClass = getattr(module, "KVReceiver")
    KVArgsClass = getattr(module, "KVArgs")
    KVBootstrapServerClass = getattr(module, "KVBootstrapServer")

    return KVManagerClass, KVSenderClass, KVReceiverClass, KVArgsClass, KVBootstrapServerClass
