# Torchvision compatibility fix for functional_tensor module
# This file helps resolve compatibility issues between different torchvision versions

import sys
import torch
import torchvision

def fix_torchvision_functional_tensor():
    """
    Fix torchvision.transforms.functional_tensor import issue
    """
    try:
        # Check if the module exists in the expected location
        import torchvision.transforms.functional_tensor
        print("torchvision.transforms.functional_tensor is available")
        return True
    except ImportError:
        print("torchvision.transforms.functional_tensor not found, applying compatibility fix...")
        
        try:
            # Create a mock functional_tensor module with the required functions
            import torchvision.transforms.functional as F
            
            class FunctionalTensorMock:
                """Mock module to replace functional_tensor"""
                
                @staticmethod
                def _get_grayscale_weights(img):
                    """Helper to create grayscale weights based on image dimensions"""
                    weights = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype)
                    return weights.view(1, 3, 1, 1) if len(img.shape) == 4 else weights.view(3, 1, 1)
                
                @staticmethod
                def _try_import_fallback(module_names, attr_name):
                    """Helper to try importing from multiple modules"""
                    for module_name in module_names:
                        try:
                            module = __import__(module_name, fromlist=[attr_name])
                            if hasattr(module, attr_name):
                                return getattr(module, attr_name)
                        except ImportError:
                            continue
                    return None
                
                @staticmethod
                def rgb_to_grayscale(img, num_output_channels=1):
                    """Convert RGB image to grayscale"""
                    if hasattr(F, 'rgb_to_grayscale'):
                        return F.rgb_to_grayscale(img, num_output_channels)
                    
                    # Fallback implementation
                    weights = FunctionalTensorMock._get_grayscale_weights(img)
                    grayscale = torch.sum(img * weights, dim=-3, keepdim=True)
                    
                    if num_output_channels == 3:
                        repeat_dims = (1, 3, 1, 1) if len(img.shape) == 4 else (3, 1, 1)
                        grayscale = grayscale.repeat(*repeat_dims)
                    
                    return grayscale
                
                @staticmethod
                def resize(img, size, interpolation=2, antialias=None):
                    """Resize function wrapper"""
                    # Try v2.functional first, then regular functional, then torch.nn.functional
                    resize_func = FunctionalTensorMock._try_import_fallback([
                        'torchvision.transforms.v2.functional',
                        'torchvision.transforms.functional'
                    ], 'resize')
                    
                    if resize_func:
                        try:
                            return resize_func(img, size, interpolation=interpolation, antialias=antialias)
                        except TypeError:
                            # Fallback for older versions without antialias parameter
                            return resize_func(img, size, interpolation=interpolation)
                    
                    # Final fallback using torch.nn.functional
                    import torch.nn.functional as torch_F
                    size = (size, size) if isinstance(size, int) else size
                    img_input = img.unsqueeze(0) if len(img.shape) == 3 else img
                    return torch_F.interpolate(img_input, size=size, mode='bilinear', align_corners=False)
                
                def __getattr__(self, name):
                    """Fallback to regular functional module"""
                    func = self._try_import_fallback([
                        'torchvision.transforms.functional',
                        'torchvision.transforms.v2.functional'
                    ], name)
                    
                    if func:
                        return func
                    
                    raise AttributeError(f"'{name}' not found in functional_tensor mock")
            
            # Create the mock module instance and monkey patch
            sys.modules['torchvision.transforms.functional_tensor'] = FunctionalTensorMock()
            print("Applied compatibility fix: created functional_tensor mock module")
            return True
            
        except Exception as e:
            print(f"Failed to create functional_tensor mock: {e}")
            return False

def apply_fix():
    """Apply the torchvision compatibility fix"""
    print(f"Torchvision version: {torchvision.__version__}")
    return fix_torchvision_functional_tensor()

if __name__ == "__main__":
    apply_fix() 
    