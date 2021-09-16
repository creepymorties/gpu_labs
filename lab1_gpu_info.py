import json
from dataclasses import dataclass
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.tools
import pprint


@dataclass
class GpuProperties:
    ComputeCapabilityMajor = pycuda.driver.device_attribute.COMPUTE_CAPABILITY_MAJOR
    ComputeCapabilityMinor = pycuda.driver.device_attribute.COMPUTE_CAPABILITY_MINOR
    MaxConstantMemory = pycuda.driver.device_attribute.TOTAL_CONSTANT_MEMORY
    SharedMemPerBlock = pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
    MaxBlockDimX = pycuda.driver.device_attribute.MAX_BLOCK_DIM_X
    MaxBlockDimY = pycuda.driver.device_attribute.MAX_BLOCK_DIM_Y
    MaxBlockDimZ = pycuda.driver.device_attribute.MAX_BLOCK_DIM_Z
    MaxGridDimX = pycuda.driver.device_attribute.MAX_GRID_DIM_X
    MaxGridDimY = pycuda.driver.device_attribute.MAX_GRID_DIM_Y
    MaxGridDimZ = pycuda.driver.device_attribute.MAX_GRID_DIM_Z
    WarpSize = pycuda.driver.device_attribute.WARP_SIZE


@dataclass
class GpuCapability:
    def __init__(self, major: int, minor: int):
        self.minor = minor
        self.major = major

    def serializable(self):
        return self.__dict__


@dataclass
class Dimensions:
    def __init__(self, x: int, y: int, z: int):
        self.z = z
        self.y = y
        self.x = x

    def serializable(self):
        return self.__dict__


@dataclass
class GpuInfo:
    def __init__(self, name: str, gpu_capability: GpuCapability, max_constant_mem: int, total_mem: int,
                 shared_mem_per_block: int, max_block_dim: Dimensions, max_grid_dim: Dimensions, warp_size: int):
        self.total_mem = total_mem
        self.warp_size = warp_size
        self.max_grid_dim = max_grid_dim
        self.max_block_dim = max_block_dim
        self.shared_mem_block = shared_mem_per_block
        self.max_constant_mem = max_constant_mem
        self.gpu_capability = gpu_capability
        self.name = name

    def serializable(self):
        return self.__dict__


def NestedSerializer(Obj):
    if hasattr(Obj, 'serializable'):
        return Obj.serializable()
    else:
        raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(Obj), repr(Obj)))


# Set device and collect all info

gpu_device = cuda.Context.get_device()
dev_all_props = cuda.Device.get_attributes(gpu_device)

# Start retrieve required values

dev_name = pycuda.tools.cuda.Device.name(gpu_device)
dev_total_memory = cuda.mem_get_info()[1]
dev_capability = GpuCapability(major=dev_all_props[GpuProperties.ComputeCapabilityMajor],
                               minor=dev_all_props[GpuProperties.ComputeCapabilityMinor])
dev_constant_mem = dev_all_props[GpuProperties.MaxConstantMemory]
dev_shared_mem_per_block = dev_all_props[GpuProperties.SharedMemPerBlock]
dev_block_max_dim = Dimensions(x=dev_all_props[GpuProperties.MaxBlockDimX],
                               y=dev_all_props[GpuProperties.MaxBlockDimY],
                               z=dev_all_props[GpuProperties.MaxBlockDimZ])

dev_grid_max_dim = Dimensions(x=dev_all_props[GpuProperties.MaxGridDimX],
                              y=dev_all_props[GpuProperties.MaxGridDimY],
                              z=dev_all_props[GpuProperties.MaxGridDimZ])

dev_warp_size = dev_all_props[GpuProperties.WarpSize]

# Create dto

full_info = GpuInfo(name=dev_name,
                    gpu_capability=dev_capability,
                    max_constant_mem=dev_constant_mem,
                    total_mem=dev_total_memory,
                    max_grid_dim=dev_grid_max_dim,
                    max_block_dim=dev_block_max_dim,
                    shared_mem_per_block=dev_shared_mem_per_block,
                    warp_size=dev_warp_size)

# Print


pprint.pprint(json.dumps(full_info, default=NestedSerializer))

# Store

with open('gpu_info.json', 'w') as f:
    json.dump(full_info, f, default=NestedSerializer)
