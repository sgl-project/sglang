import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass
import uuid
import json

import torch
import numpy as np

logger = logging.getLogger(__name__)


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
	hasher = hashlib.sha256()

	if prior_hash:
		hasher.update(bytes.fromhex(prior_hash))

	for t in token_ids:
		hasher.update(t.to_bytes(4, byteorder="little", signed=False))

	return hasher.hexdigest()

def get_hash_str_mooncake(
	prefix_block_key: str, current_page_ids: List, local_rank: int
):
	prefix_str = ""
	if prefix_block_key:
		if len(prefix_block_key):
			prefix_str = hashlib.sha256(prefix_block_key.encode()).hexdigest()
	current_token_ids_bytes = np.array(current_page_ids).tobytes()
	current_hash_object = hashlib.sha256(current_token_ids_bytes)
	current_hash_hex = current_hash_object.hexdigest()
	return f"{prefix_str}_{int(current_hash_hex[:16], 16)}_{local_rank}"


# todo: batch API for better performance


class HiCacheStorage(ABC):
	"""
	HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
	It abstracts the underlying storage mechanism, allowing different implementations to be used.
	"""

	@abstractmethod
	def get(
		self, 
		key, 
		target_location: Optional[Any] = None,
		target_sizes: Optional[Any] = None
	) -> torch.Tensor | None:
		"""
		Retrieve the value associated with the given key.
		Returns None if the key does not exist.
		"""
		pass

	@abstractmethod
	def set(self, 
		 	key, 
		 	value: Optional[Any] = None, 
		 	target_location: Optional[Any] = None,
			target_sizes: Optional[Any] = None) -> bool:
		"""
		Store the value associated with the given key.
		Returns True if the operation was successful, False otherwise.
		"""
		pass

	@abstractmethod
	def delete(self, key) -> None:
		"""
		Delete the value associated with the given key.
		"""
		pass

	@abstractmethod
	def exists(self, key) -> bool | dict:
		"""
		Check if the key exists in the storage.
		Returns True if the key exists, False otherwise.
		"""
		pass

	@abstractmethod
	def clear(self) -> None:
		"""
		Clear all entries in the storage.
		"""
		pass


class HiCacheFile(HiCacheStorage):

	def __init__(self, file_path: str = "/tmp/hicache"):
		self.file_path = file_path
		if not os.path.exists(self.file_path):
			os.makedirs(self.file_path)
			logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

	def get(
		self, 
		key, 
		target_location: Optional[Any] = None,
		target_sizes: Optional[Any] = None
	) -> torch.Tensor | None:
		tensor_path = f"{self.file_path}/{key}.bin"
		try:
			# todo: fixing the target_location logic to enable in-place loading
			loaded_tensor = torch.load(tensor_path)
			if isinstance(loaded_tensor, torch.Tensor):
				return loaded_tensor
			else:
				logger.error(f"Loaded data for key {key} is not a tensor.")
				return None
		except FileNotFoundError:
			return None

	def set(self,
			key, 
		 	value: Optional[Any] = None, 
		 	target_location: Optional[Any] = None,
			target_sizes: Optional[Any] = None) -> bool:
		tensor_path = f"{self.file_path}/{key}.bin"
		if self.exists(key):
			logger.warning(f"Key {key} already exists. Skipped.")
			return True
		try:
			torch.save(value, tensor_path)
			return True
		except Exception as e:
			logger.error(f"Failed to save tensor {key}: {e}")
			return False

	def delete(self, key: str) -> None:
		tensor_path = f"{self.file_path}/{key}.bin"
		try:
			os.remove(tensor_path)
		except FileNotFoundError:
			logger.warning(f"Key {key} does not exist. Cannot delete.")
			return

	def exists(self, key) -> bool | dict:
		tensor_path = f"{self.file_path}/{key}.bin"
		return os.path.exists(tensor_path)

	def clear(self) -> None:
		try:
			for filename in os.listdir(self.file_path):
				file_path = os.path.join(self.file_path, filename)
				if os.path.isfile(file_path):
					os.remove(file_path)
			logger.info("Cleared all entries in HiCacheFile storage.")
		except Exception as e:
			logger.error(f"Failed to clear HiCacheFile storage: {e}")

@dataclass
class MooncakeStoreConfig:
	local_hostname: str
	metadata_server: str
	global_segment_size: int
	local_buffer_size: int
	protocol: str
	device_name: str
	master_server_address: str

	@staticmethod
	def from_file(file_path: str) -> "MooncakeStoreConfig":
		"""Load the config from a JSON file."""
		with open(file_path) as fin:
			config = json.load(fin)
		return MooncakeStoreConfig(
			local_hostname=config.get("local_hostname"),
			metadata_server=config.get("metadata_server"),
			global_segment_size=config.get(
				"global_segment_size"
			),
			local_buffer_size=config.get(
				"local_buffer_size"
			),
			protocol=config.get("protocol", "tcp"),
			device_name=config.get("device_name", ""),
			master_server_address=config.get("master_server_address"),
		)

	@staticmethod
	def load_from_env() -> "MooncakeStoreConfig":
		"""Load config from a file specified in the environment variable."""
		config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
		if config_file_path is None:
			raise ValueError(
				"The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
			)
		return MooncakeStoreConfig.from_file(config_file_path)


class MooncakeStore(HiCacheStorage):
	def __init__(self):
		try:
			from mooncake.store import MooncakeDistributedStore
		except ImportError as e:
			raise ImportError(
				"Please install mooncake by following the instructions at "
				"https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
				"to run vLLM with MooncakeConnector."
			) from e

		try:
			self.store = MooncakeDistributedStore()
			self.config = MooncakeStoreConfig.load_from_env()
			logger.info("Mooncake Configuration loaded successfully.")

			setup_code = self.store.setup(
				self.config.local_hostname,
				self.config.metadata_server,
				self.config.global_segment_size,
				self.config.local_buffer_size,
				self.config.protocol,
				self.config.device_name,
				self.config.master_server_address,
			)
			assert setup_code == 0
			logger.info("Connect to Mooncake store successfully.")
			self.warmup()
			logger.info("Mooncake store warmup successfully.")

		except ValueError as e:
			logger.error("Configuration loading failed: %s", e)
			raise
		except Exception as exc:
			logger.error("An error occurred while loading the configuration: %s", exc)
			raise
		
	def warmup(self):
		warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
		# 10 MB
		warmup_value = bytes(10 * 1024 * 1024)
		self.store.put(warmup_key, warmup_value)
		assert self.store.is_exist(warmup_key) == 1
		self.store.get(warmup_key)
		self.store.remove(warmup_key)

	def register_buffer(
		self,
		buffer: torch.Tensor
	) -> None:
		try:
			buffer_ptr = buffer.data_ptr()
			buffer_size = buffer.numel() * buffer.element_size()
			self.store.register_buffer(buffer_ptr, buffer_size)
		except TypeError as err:
			logger.error("Failed to register buffer to Mooncake Store: %s", err)
			raise TypeError("Mooncake Store Register Buffer Error.") from err


	def set(
		self, 
		key, 
		value: Optional[Any] = None, 
		target_location: Optional[List[int]] = None,
		target_sizes: Optional[List[int]] = None) -> bool:
		assert len(key) == len(target_location) == len(target_sizes)
		if len(key) == 0:
			return

		for i in range(len(key)):
			if (key[i] is None
				or target_location[i] is None
				or target_sizes[i] is None):
				return

		self._put_batch_zero_copy_impl(key, target_location, target_sizes)

	def get(
		self, 
		key, 
		target_location: Optional[Any] = None,
		target_sizes: Optional[Any] = None
	) -> torch.Tensor | None:
		assert len(key) == len(target_location) == len(target_sizes)
		if len(key) == 0:
			return

		for i in range(len(key)):
			if (key[i] is None
				or target_location[i] is None
				or target_sizes[i] is None):
				return

		return self._get_batch_zero_copy_impl(key, target_location, target_sizes)
	
	def exists(self, keys) -> bool | dict:
		_keys = []
		for key in keys:
			if key is None:
				return None
			# Since mooncake store is stored in layer by layer,
			# only the first layer is checked here.
			_keys.append(f"{key}_0_k")
		result = {k: v for k, v in zip(keys, self.store.batch_is_exist(_keys))}
		return result

	def delete(self, key) -> None:
		raise(NotImplementedError)
	

	def close(self):
		# MooncakeDistributedStore will automatically call the destructor, so
		# it is unnecessary to close it manually.
		pass

	def clear(self) -> None:
		raise(NotImplementedError)

	def _put_batch_zero_copy_impl(
		self,
		key_strs: List[str],
		buffer_ptrs: List[int],
		buffer_sizes: List[int]
	) -> None:
		try:
			self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)
		except TypeError as err:
			logger.error("Failed to put value to Mooncake Store: %s", err)
			raise TypeError("Mooncake Store Put Type Error.") from err

	def _get_batch_zero_copy_impl(
		self,
		key_strs: List[str],
		buffer_ptrs: List[int],
		buffer_sizes: List[int]
	) -> None:
		try:
			self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
		except TypeError as err:
			logger.error("Failed to get value from Mooncake Store: %s", err)
			raise TypeError("Mooncake Store Get Type Error.") from err