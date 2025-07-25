import logging

logger = logging.getLogger(__name__)

class NixlBackendSelection:
    """Handles NIXL backend selection and creation."""
    
    def __init__(self, file_plugin: str = "auto"):
        self.file_plugin = file_plugin
    
    def create_backend(self, agent) -> bool:
        """Create the appropriate NIXL backend based on configuration."""
        try:
            # Get available plugins
            plugin_list = agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")
            
            # Select backend based on file_plugin setting
            if self.file_plugin == "auto":
                if "GDS_MT" in plugin_list:
                    backend_name = "GDS_MT"
                elif "GDS" in plugin_list:
                    backend_name = "GDS"
                elif "POSIX" in plugin_list:
                    backend_name = "POSIX"
                else:
                    logger.warning("No suitable NIXL backend found, using POSIX")
                    backend_name = "POSIX"
            else:
                backend_name = self.file_plugin
            
            # Create the selected backend
            if backend_name in plugin_list:
                agent.create_backend(backend_name)
                logger.debug(f"Created NIXL backend: {backend_name}")
                return True
            else:
                logger.error(f"Backend {backend_name} not available in plugins: {plugin_list}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create NIXL backend: {e}")
            return False 
