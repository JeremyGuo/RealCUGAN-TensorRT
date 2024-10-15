import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import torch
import threading

cuda_init_lock = threading.Lock()
cuda_init_flag = False
cuda_execution_lock = threading.Lock()

def initCUDA():
    global cuda_init_flag
    global cuda_init_lock
    with cuda_init_lock:
        if not cuda_init_flag:
            cuda.init()
            cuda_init_flag = True

class TRTModule(torch.nn.Module):
    def __init__(self, engine_path, device_index):
        initCUDA()
        super(TRTModule, self).__init__()
        self.ctx = cuda.Device(device_index).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape) * dtype.itemsize
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print(f"Input: {name}, {shape}, {dtype}")
                self.inputs.append({"name": name,
                                    "host": cuda.pagelocked_empty(tuple(shape), dtype),
                                    "device": cuda.mem_alloc(size), "shape": shape, "dtype": dtype})
                # self.inputs.append({"name": name, "shape": shape, "dtype": dtype})
            else:
                print(f"Output: {name}, {shape}, {dtype}")
                self.outputs.append({"name": name,
                                     "host": cuda.pagelocked_empty(tuple(shape), dtype),
                                     "device": cuda.mem_alloc(size), "shape": shape, "dtype": dtype})
        
    def forward(self, *inputs):
        self.ctx.push()

        # Transfer input data to device
        for i, inp in enumerate(inputs):
            np.copyto(self.inputs[i]["host"], inp)
            cuda.memcpy_htod_async(self.inputs[i]["device"], self.inputs[i]["host"], self.stream)
        
        # Run inference
        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], inp["device"])
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], out["device"])

        with cuda_execution_lock:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()

            
        # Transfer predictions back
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            outputs.append(torch.from_numpy(out["host"].reshape(out["shape"])))
    
        # Synchronize the stream
        self.stream.synchronize()
        self.ctx.pop()
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    def __del__(self):
        del self.inputs
        del self.outputs
        del self.stream
        del self.context
        del self.engine
        cuda.Context.pop()