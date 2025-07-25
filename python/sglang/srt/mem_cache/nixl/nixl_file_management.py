import os
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class NixlFileManagement:
    """Handles file operations for NIXL storage."""
    
    def __init__(self, base_path: str = "/tmp/hicache_nixl"):
        self.base_path = base_path
        self.registered_files: Dict[str, Tuple[int, any]] = {}  # file_path -> (fd, nixl_descs)
    
    def ensure_directory_exists(self) -> bool:
        """Ensure the base directory exists."""
        try:
            os.makedirs(self.base_path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {self.base_path}: {e}")
            return False
    
    def get_file_path(self, key: str) -> str:
        """Get the file path for a given key."""
        return os.path.join(self.base_path, f"{key}.bin")
    
    def create_file(self, file_path: str) -> bool:
        """Create a new file and return success status."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create empty file
            with open(file_path, 'wb') as f:
                pass
            
            logger.debug(f"Created file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            return False
    
    def open_file(self, file_path: str) -> Optional[int]:
        """Open a file and return file descriptor."""
        try:
            fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
            logger.debug(f"Opened file {file_path} with fd: {fd}")
            return fd
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return None
    
    def close_file(self, fd: int) -> bool:
        """Close a file descriptor."""
        try:
            os.close(fd)
            return True
        except Exception as e:
            logger.error(f"Failed to close fd {fd}: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def cleanup_files(self):
        """Clean up all registered files."""
        for file_path, (fd, _) in self.registered_files.items():
            try:
                self.close_file(fd)
                self.delete_file(file_path)
            except Exception as e:
                logger.error(f"Failed to cleanup file {file_path}: {e}")
        self.registered_files.clear() 