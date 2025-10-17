##
##
import torch
import torch.cuda.nvtx as nvtx


class PytHooks(object):
    """This module contains all the code needed to enable forward hooks in a pytorch network.

    To register the hooks for a given network, the user needs to instantiate a PytHook object.
    Then call the register_hooks method.

    Example:

        my_hook = PytHook()
        my_hook.register_hooks(my_network_model)
    """

    def __init__(self, debug=False):
        """Initialize module variables

        Args:
            None:

        Returns:
            None:

        Raises:
            None:
        """
        super().__init__()
        self.module_to_name_map = {}
        self.out_tensor_to_name_stack = {}
        self.iteration = 0
        self.debug = debug

    def module_bwd_hook(self, module, grad_input, grad_output):
        """Callback function for the backward hook of a module

        Args:
            module: Pointer to the module object.
            grad_input: Input tensor used to compute gradients.
            grad_output: Output tensor of the resulting gradient.

        Returns:
            None:

        Raises:
            None:
        """
        if self.debug:
            print(
                f"BWD hook module:{module} addr:{hex(id(module))} "
                "grad_input:{grad_input} grad_output:{grad_output} "
                "params:{module.parameters}"
            )
        return

    def param_bwd_hook(self, grad):
        """Callback function for the backward hook of a tensor

        Args:
            grad: Tensor which contains the resulting gradient.

        Returns:
            None:

        Raises:
            None:
        """
        if self.debug:
            print(
                f"Param BWD Hook grad tensor type:{type(grad)} val:{grad} "
                "name:{grad.name} grads:{grad.grad} cdata:{hex(grad._cdata)}"
            )
        return

    @staticmethod
    def format_module_name(module_name):
        """String formatting function

        Adds '' around the module name string.

        Args:
            module_name: String name of the module

        Returns:
            name: String with '' around module name

        Raises:
            None:
        """
        name = "'{}'".format(module_name)
        return name

    def get_input_tensors(self, tensor_obj):
        """Returns input tensors in list format

        The in_tensor_seq can be any number of different data types
        This function puts them in list format.

        Args:
            tensor_obj: Could be a Tensor or an iterator type that contains Tensors
            prefix: String name to assign to the Tensor

        Returns:
            None:

        Raises:
            None:
        """
        tensor_list = []
        ## We don't know if list or seq is just a list of tensors
        ## or a list of sequences of tensors.  So need to descend to the
        ## lowest level and build a list of tensors from there
        if isinstance(tensor_obj, list) or isinstance(tensor_obj, tuple):
            for ten in tensor_obj:
                tmp_list = self.get_input_tensors(ten)
                tensor_list.extend(tmp_list)
        elif isinstance(tensor_obj, torch.Tensor):
            tensor_ptr = hex(id(tensor_obj))
            if self.debug:
                print(
                    f"Input -> shape {list(tensor_obj.size())} pointer:" "{tensor_ptr}"
                )
            tensor_list.append(tensor_obj)
        return tensor_list

    @staticmethod
    def print_tensor(tensor_obj, prefix, tensor_list=[]):
        """Descends iterators that contains Tensors and prints the Tensor

        Recursive function that descends iterator type arguments until
        it finds a Tensor object.

        Args:
            tensor_obj: Could be a Tensor or an iterator type that contains Tensors
            prefix: String name to assign to the Tensor

        Returns:
            None:

        Raises:
            None:
        """
        tensor_dims = []
        tensor_ptr = None
        if isinstance(tensor_obj, list) or isinstance(tensor_obj, tuple):
            for ten in tensor_obj:
                tensor_list = PytHooks.print_tensor(ten, prefix, tensor_list)
        elif isinstance(tensor_obj, torch.Tensor):
            tensor_ptr = hex(id(tensor_obj))
            tensor_dims = list(tensor_obj.size())
            tensor_list.append(tensor_dims)
        return tensor_list

    def clear_saved_tensors(self):
        """Clears the saved output tensors at end of each iteration"""
        if self.debug:
            print("Clearing saved output tensors")
        self.out_tensor_to_name_stack.clear()
        return

    def push_output_tensor(self, tensor_ptr, module_name):
        """Each output tensor is stored in a stack that maps to the module hierarchy

        Args:
            tensor_ptr: Tensor address is the key to lookup the stack of module names
                that the tensor originated from.

            module_name: The current module that output this tensor

        Raises:
            None
        Returns:
            None

        """
        self.out_tensor_to_name_stack[tensor_ptr] = module_name
        if self.debug:
            print(f"Out tensor:{tensor_ptr} from module:{module_name} pushed")
        return

    def pop_output_tensor(self, tensor_ptr):
        module_name = None
        if tensor_ptr in self.out_tensor_to_name_stack:
            module_name = self.out_tensor_to_name_stack[tensor_ptr]
        return module_name

    def process_layer_params(self, module_obj):
        """Extract the static parameters from each of the nn.Layer types

        Args:
            module_obj(class): Module state data structure.

        Returns:
            param_info(dict): Parameter meta_data for the given op.

        Raises:
            None

        """
        param_info = {}
        ## Look for layer parameters specific to each layer type
        if (
            isinstance(module_obj, torch.nn.Conv1d)
            or isinstance(module_obj, torch.nn.Conv2d)
            or isinstance(module_obj, torch.nn.Conv3d)
        ):
            conv_params = {}
            conv_params["in_chan"] = module_obj.in_channels
            conv_params["out_chan"] = module_obj.out_channels
            conv_params["filter_dim"] = module_obj.kernel_size
            conv_params["stride"] = module_obj.stride
            conv_params["padding"] = module_obj.padding
            conv_params["dilation"] = module_obj.dilation
            conv_params["transposed"] = module_obj.transposed
            conv_params["output_padding"] = module_obj.output_padding
            conv_params["groups"] = module_obj.groups
            conv_params["padding_mode"] = module_obj.padding_mode
            ## @@@ Add these to the nvtx marker
            param_info = conv_params
        elif (
            isinstance(module_obj, torch.nn.ConvTranspose1d)
            or isinstance(module_obj, torch.nn.ConvTranspose2d)
            or isinstance(module_obj, torch.nn.ConvTranspose3d)
        ):
            convtranspose_params = {}
            convtranspose_params["in_chan"] = module_obj.in_channels
            convtranspose_params["out_chan"] = module_obj.out_channels
            convtranspose_params["filter_dim"] = module_obj.kernel_size
            convtranspose_params["stride"] = module_obj.stride
            convtranspose_params["padding"] = module_obj.padding
            convtranspose_params["dilation"] = module_obj.dilation
            convtranspose_params["transposed"] = module_obj.transposed
            convtranspose_params["output_padding"] = module_obj.output_padding
            convtranspose_params["groups"] = module_obj.groups
            convtranspose_params["padding_mode"] = module_obj.padding_mode
            ## @@@ Add these to the nvtx marker
            param_info = convtranspose_params
        elif (
            isinstance(module_obj, torch.nn.MaxPool1d)
            or isinstance(module_obj, torch.nn.MaxPool2d)
            or isinstance(module_obj, torch.nn.MaxPool3d)
        ):

            def _handle_int_or_tuple(parameter):
                if type(parameter) is tuple:
                    return list(parameter)
                elif type(parameter) is int:
                    return [parameter, parameter]

            pooling_params = {}
            pooling_params["filter_dim"] = _handle_int_or_tuple(module_obj.kernel_size)
            pooling_params["stride"] = _handle_int_or_tuple(module_obj.stride)
            pooling_params["padding"] = _handle_int_or_tuple(module_obj.padding)
            pooling_params["dilation"] = _handle_int_or_tuple(module_obj.dilation)
            param_info = pooling_params
        elif (
            isinstance(module_obj, torch.nn.AvgPool1d)
            or isinstance(module_obj, torch.nn.AvgPool2d)
            or isinstance(module_obj, torch.nn.AvgPool3d)
        ):
            pooling_params = {}
            pooling_params["filter_dim"] = [
                module_obj.kernel_size,
                module_obj.kernel_size,
            ]
            pooling_params["stride"] = [module_obj.stride, module_obj.stride]
            pooling_params["padding"] = [module_obj.padding, module_obj.padding]
            pooling_params["ceil_mode"] = module_obj.ceil_mode
            pooling_params["count_include_pad"] = module_obj.count_include_pad
            param_info = pooling_params
        elif (
            isinstance(module_obj, torch.nn.AdaptiveAvgPool1d)
            or isinstance(module_obj, torch.nn.AdaptiveAvgPool2d)
            or isinstance(module_obj, torch.nn.AdaptiveAvgPool3d)
        ):
            pooling_params = {}
            pooling_params["output_size"] = [
                module_obj.output_size,
                module_obj.output_size,
            ]
            param_info = pooling_params
        elif isinstance(module_obj, torch.nn.Linear):
            param_info["in_features"] = module_obj.in_features
            param_info["out_features"] = module_obj.out_features
        elif (
            isinstance(module_obj, torch.nn.BatchNorm1d)
            or isinstance(module_obj, torch.nn.BatchNorm2d)
            or isinstance(module_obj, torch.nn.BatchNorm3d)
        ):
            param_info["num_features"] = module_obj.num_features
            param_info["epsilon"] = module_obj.eps
            param_info["momentum"] = module_obj.momentum
        elif isinstance(module_obj, torch.nn.ReLU):
            param_info["in_place"] = module_obj.inplace
        elif isinstance(module_obj, torch.nn.Dropout):
            prob = module_obj.p
            param_info["in_place"] = module_obj.inplace
        elif isinstance(module_obj, torch.nn.Embedding):
            param_info["num_embeddings"] = module_obj.num_embeddings
            param_info["embedding_dim"] = module_obj.embedding_dim
        elif isinstance(module_obj, torch.nn.ReflectionPad1d):
            # keeping this limited to ReflectionPad1d for now because the normal
            # torch.nn.function.pad layer is more complicated in how it handles params
            param_info["padding"] = module_obj.padding
        elif isinstance(
            module_obj,
            (
                torch.nn.Upsample,
                torch.nn.UpsamplingNearest2d,
                torch.nn.UpsamplingBilinear2d,
            ),
        ):
            param_info["scale_factor"] = module_obj.scale_factor
        elif isinstance(module_obj, torch.nn.LSTM):
            param_info = self.process_lstm_layer_params(module_obj)

        if self.debug:
            print(f"Module: {type(module_obj).__name__} Params: {param_info}")
        return param_info

    def process_lstm_layer_params(self, module_obj):
        param_info = {}
        param_info["cell"] = "lstm"
        param_info["hidden_size"] = module_obj.hidden_size
        param_info["batch_first"] = module_obj.batch_first
        param_info["bidirectional"] = module_obj.bidirectional
        param_info["input_size"] = module_obj.input_size
        param_info["num_layers"] = module_obj.num_layers
        if self.debug:
            print(f"Module: LSTM Params: {param_info}")
        return param_info

    def module_fwd_hook(self, module_obj, in_tensor, out_tensor):
        """Callback function that ends the NVTX marker

        Records the module name and tensor information
        Called after the module executes the forward method.

        Args:
            module_obj: Pointer to the module object
            in_tensor: Input tensor or list of tensors
            out_tensor: Output tensor of the resulting forward operator

        Returns:
            None:

        Raises:
            None:
        """
        nvtx.range_pop()
        module_name = self.module_to_name_map[module_obj]
        module_params = module_obj.named_parameters(prefix=module_name, recurse=False)

        if self.debug:
            print(f"FWD hook module {module_name}")
        out_tensor_list = PytHooks.print_tensor(out_tensor, "Output")
        # self.push_output_tensor(out_tensor_ptr, module_name)
        if module_name == "'top'":
            self.iteration = self.iteration + 1
            self.clear_saved_tensors()
            if self.debug:
                print(f"Completed {self.iteration} iterations")

        return

    def module_fwd_pre_hook(self, module_obj, in_tensor):
        """Creates an NVTX marker with the module name in it.

        This function is called before the module executes

        Args:
            module_obj: Module object data structure - used to get unique module name
            in_tensor: Input tensor data structure

        Returns:
            None

        Raises:
            None
        """

        marker_dict = {}
        module_name = self.module_to_name_map[module_obj]
        module_params = module_obj.named_parameters(recurse=False)
        if self.debug:
            print(f"FWD Pre hook module:{module_name}")
        marker_dict["Module"] = module_name

        ## Get trainable parameters like weights and bias
        for idx, (param_name, param_obj) in enumerate(module_params):
            if idx == 0:
                marker_dict["TrainableParams"] = {}
            marker_dict["TrainableParams"][param_name] = list(param_obj.size())
            if self.debug:
                print(f"Param {param_name} value {list(param_obj.size())}")

        in_tensor_list = PytHooks.print_tensor(in_tensor, "Input", tensor_list=[])
        if in_tensor_list:
            marker_dict["Inputs"] = in_tensor_list
            if self.debug:
                print("Input Tensor List-> {in_tensor_list}")

        param_info = self.process_layer_params(module_obj)
        if param_info:
            marker_dict["StaticParams"] = param_info

        nvtx.range_push("{}".format(marker_dict))

        return

    def register_hooks(self, network_model, module_prefix="top"):
        """User level function that activates all the hooks

        The user needs to call this method from the network source code
        The code descends all the modules in the network and registers their
        respective hooks.

        Args:
            network_model: Model object for the network
            module_prefix: (default: top)

        Returns:
            None

        Raises:
            Exception if a module instance is reused
        """
        for name, module in network_model.named_modules(prefix=module_prefix):
            if self.debug:
                print(f"Module Name:{name} addr:{hex(id(module))}")
            module.register_forward_pre_hook(self.module_fwd_pre_hook)
            module.register_forward_hook(self.module_fwd_hook)
            if module not in self.module_to_name_map:
                self.module_to_name_map[module] = name
            else:
                raise Exception("Module instance {} is not unique ".format(module))
        return


class ANNAProfControl(object):
    """This class controls the start and stop off the profiler."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def profiler_start():
        """
        This method starts the profiler.
        """
        torch.cuda.cudart().cudaProfilerStart()
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    @staticmethod
    def profiler_stop(exit_program=True):
        """
        This method stops the profiler.

        Args:
            exit_program: (default: True) exits the program after profiling is stopped
        """
        torch.cuda.cudart().cudaProfilerStop()
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)

        if exit_program:
            exit(0)
