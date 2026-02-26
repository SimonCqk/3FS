# GPU Direct RDMA (GDR) Support for 3FS

This document describes the GPU Direct RDMA (GDR) implementation for 3FS distributed filesystem, enabling direct data transfers between storage and GPU memory without CPU memory copies.

## Overview

In distributed inference scenarios with KV Cache, the traditional data path involves:
1. Storage Service → RDMA → Host Memory
2. Host Memory → cudaMemcpy → GPU Memory

With GDR enabled:
1. Storage Service → RDMA → GPU Memory (direct)

This eliminates one memory copy, reducing latency and improving throughput for GPU-accelerated workloads.

## Architecture

### Components

1. **AcceleratorMemory Module** (`AcceleratorMemory.h/cc`)
   - Core accelerator memory registration with IB devices
   - GDRManager singleton for global state
   - Memory region caching for efficiency
   - Vendor-neutral naming supports future AMD/Intel accelerators

2. **Unified usrbio API** (`hf3fs_usrbio.h`)
   - Unified C API for both host and accelerator memory I/O
   - Automatic memory type detection
   - Device hint via negative numa parameter: `numa = -(device_id + 1)`
   - Transparent fallback to host memory when GDR unavailable

3. **Device API internals** (`UsrbIoGdr.cc`)
   - Legacy GPU-specific API for backward compatibility
   - All functions marked deprecated
   - Wraps unified API internally

4. **RDMABufAccelerator** (`RDMABufAccelerator.h/cc`)
   - Accelerator-aware RDMA buffer implementation
   - Unified interface with regular RDMABuf
   - Accelerator memory pool support

5. **AcceleratorShmBuf** (`AcceleratorShm.h/cc`)
   - Accelerator shared memory buffer for fuse integration
   - IPC channel for accelerator handle transfer
   - Storage client integration

6. **Accelerator Memory Bridge** (`AcceleratorMemoryBridge.h/cc`)
   - Cross-process accelerator memory import mechanisms
   - CUDA IPC support
   - Handles CUDA context ownership issues

### Data Flow

```
Inference Engine Process              Fuse Daemon Process
┌─────────────────────────┐          ┌─────────────────────────┐
│  1. Create Device IOV   │          │                         │
│  hf3fs_iovcreate_device │          │                         │
│                         │          │                         │
│  2. IPC Handle (auto)   │──────────▶ 3. Import via symlink  │
│     (exported in create)│   fuse   │    (gdr:// URI)        │
│                         │          │                         │
│  4. Submit I/O          │          │ 5. Register for RDMA   │
│     hf3fs_prep_io()     │──────────▶    GDRManager          │
│     hf3fs_submit_ios()  │          │                         │
│                         │          │ 6. RDMA Read to GPU    │
│  7. Wait for completion │◀──────────│    IBSocket::rdmaRead │
│     hf3fs_wait_for_ios()│          │                         │
└─────────────────────────┘          └─────────────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────────────┐
                                    │    Storage Service      │
                                    │    (RDMA Target)        │
                                    └─────────────────────────┘
```

## Requirements

### Hardware
- NVIDIA GPU with GPUDirect RDMA support (Kepler or newer)
- Mellanox/NVIDIA ConnectX-4 or newer NIC
- PCIe topology allowing GPU-NIC peer access

### Software
- Linux kernel 4.x+
- CUDA Toolkit 11.0+
- nvidia_peermem kernel module
- MLNX_OFED drivers

### Kernel Modules
```bash
# Load nvidia_peermem for GDR support
modprobe nvidia_peermem

# Verify
lsmod | grep nvidia_peermem
```

## Build Integration

### Enabling GDR Support

GDR support is optional and disabled by default. To enable:

```bash
# Configure with GDR support
cmake -B build -DENABLE_GDR=ON -DSHUFFLE_METHOD=stdshuffle

# Build
cmake --build build
```

The build system will:
1. Search for CUDA Toolkit
2. Define `HF3FS_GDR_ENABLED` compile flag
3. Link CUDA libraries to relevant targets

### Build Requirements
- CUDA Toolkit 11.0+ installed
- `nvcc` and CUDA headers available

## API Reference

### Unified API (Recommended)

The unified API in `hf3fs_usrbio.h` automatically handles both host and accelerator memory. This is the recommended approach for new code.

#### Device Hint via Negative NUMA

Use the `_device` API variants for accelerator memory:
- `hf3fs_iovcreate(numa >= 0)`: Host memory on specified NUMA node
- `hf3fs_iovcreate_device(device_id)`: Accelerator device memory, falls back to host if GDR unavailable
- `hf3fs_iovopen_device(device_id)`: Reopen existing device IOV (requires GDR)
- `hf3fs_iovwrap_device(device_ptr, device_id)`: Wrap external device memory (requires GDR)

#### Device API Example

```c
#include "lib/api/hf3fs_usrbio.h"

// Create IOV on GPU device 0 (or fallback to host if GDR unavailable)
struct hf3fs_iov iov;
int ret = hf3fs_iovcreate_device(&iov, "/mnt/hf3fs", size, 0, 0);

// Check memory type
enum hf3fs_mem_type mem_type = hf3fs_iov_mem_type(&iov);
if (mem_type == HF3FS_MEM_DEVICE) {
    int device_id = hf3fs_iov_device_id(&iov);
    printf("Using GPU device %d\n", device_id);
} else {
    printf("Using host memory (GDR unavailable)\n");
}

// Use for I/O (same as regular usrbio)
struct hf3fs_ior ior;
hf3fs_iorcreate4(&ior, "/mnt/hf3fs", 64, true, 0, 0, 0, 0);
hf3fs_prep_io(&ior, &iov, true, iov.base, fd, 0, size, NULL);
hf3fs_submit_ios(&ior);
hf3fs_wait_for_ios(&ior, &cqe, 1, 1, NULL);

// Cleanup
hf3fs_iovdestroy(&iov);
hf3fs_iordestroy(&ior);
```

#### Wrapping Existing Device Memory

Use this when you have GPU memory from an external source (e.g., deep learning framework):

```c
void* gpu_ptr;  // e.g., from PyTorch, TensorFlow, or cudaMalloc
uint8_t uuid[16];
generate_uuid(uuid);

struct hf3fs_iov iov;
int ret = hf3fs_iovwrap_device(&iov, gpu_ptr, uuid, "/mnt/hf3fs",
                               buffer_size, 0, gpu_device_id);
// Returns -ENOTSUP if GDR is unavailable

// Use for I/O (same as above)
hf3fs_prep_io(&ior, &iov, true, gpu_ptr, fd, offset, len, NULL);
hf3fs_submit_ios(&ior);

// Cleanup - only releases RDMA registration, does NOT free GPU memory
hf3fs_iovdestroy(&iov);
```

#### Cross-Process Device Memory Sharing

IPC handles are exported automatically when creating device IOVs via `hf3fs_iovcreate_device`.

```c
// Create device IOV - IPC handle exported automatically
struct hf3fs_iov iov;
hf3fs_iovcreate_device(&iov, "/mnt/hf3fs", size, 0, 0);  // device 0
// IPC handle is available for cross-process sharing via fuse namespace
```

### Utility Functions

#### Unified API (Recommended)

```c
// Check memory type
enum hf3fs_mem_type mem_type = hf3fs_iov_mem_type(&iov);
if (mem_type == HF3FS_MEM_DEVICE) {
    // Get device ID
    int device_id = hf3fs_iov_device_id(&iov);
    printf("Accelerator iov on device %d\n", device_id);
} else if (mem_type == HF3FS_MEM_HOST) {
    printf("Host memory iov\n");
}

// Synchronize accelerator memory for coherency (usually automatic)
// direction: 0 = before RDMA (accelerator writes visible to RDMA)
//           1 = after RDMA (RDMA writes visible to accelerator)
hf3fs_iovsync(&iov, 0);
```

#### Deprecated GPU API

```c
// DEPRECATED: Use hf3fs_iov_mem_type instead
if (hf3fs_iov_is_gpu(&iov)) {
    // DEPRECATED: Use hf3fs_iov_device_id instead
    int device_id = hf3fs_iov_gpu_device(&iov);
    printf("GPU iov on device %d\n", device_id);
}

// DEPRECATED: Use hf3fs_iovsync instead
hf3fs_iovsync_gpu(&iov, 0);
```

## Configuration

### Environment Variables

- `HF3FS_USRBIO_LIB_LOG`: Log level for usrbio library (e.g., "INFO", "WARN", "DBG")
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are available

## Performance Considerations

1. **Memory Alignment**: GPU memory should be 256-byte aligned for optimal GDR performance.

2. **PCIe Topology**: Best performance when GPU and NIC are on the same PCIe switch.
   ```bash
   # Check topology
   nvidia-smi topo -m
   ```

3. **Transfer Size**: GDR is most beneficial for larger transfers (>64KB).

4. **Registration Caching**: GPU memory regions are cached internally for efficiency.

5. **Synchronization**: The nvidia_peermem driver handles coherency automatically in most cases.

## Troubleshooting

### Common Issues

1. **"GDR not available"**
   ```bash
   # Check nvidia_peermem is loaded
   modprobe nvidia_peermem
   lsmod | grep nvidia_peermem

   # Check RDMA devices
   ibv_devinfo
   ```

2. **"Failed to register GPU memory with RDMA"**
   - Check that nvidia_peermem is loaded
   - Verify GPU-NIC PCIe topology allows peer access
   - Check ibv_devinfo for device status

3. **"CUDA IPC import failed"**
   - Ensure CUDA is available in the importing process
   - Check CUDA_VISIBLE_DEVICES is configured correctly
   - Verify both processes can access the same GPU

4. **Performance lower than expected**
   - Check `nvidia-smi topo -m` for GPU-NIC distance
   - Use `ibv_devinfo -v` to verify RDMA capabilities
   - Profile with `nvprof` or `nsys`

## Implementation Status

### Completed (v1.0)
- ✅ Core accelerator memory registration with IB devices
- ✅ Vendor-neutral naming (AcceleratorMemory, AcceleratorMemoryBridge, RDMABufAccelerator)
- ✅ Unified API in hf3fs_usrbio.h with automatic memory type detection
- ✅ Device hint via negative numa parameter
- ✅ Automatic fallback to host memory when GDR unavailable
- ✅ Deprecated GPU-specific API for backward compatibility
- ✅ Automatic IPC handle export for cross-process sharing
- ✅ Environment variable controls (HF3FS_GDR_ENABLED, HF3FS_GDR_FALLBACK_MODE)
- ✅ ShmBuf extended with accelerator memory detection
- ✅ Integration tests and documentation

### Future Enhancements

1. **AMD ROCm Support**: Extend to AMD GPUs using ROCm runtime
   - Add ROCm memory detection in detectMemoryType()
   - Implement ROCm-specific memory registration
   - Test with AMD Instinct accelerators

2. **Intel oneAPI Support**: Extend to Intel GPUs using Level Zero
   - Add Level Zero memory detection
   - Implement Intel GPU memory registration
   - Test with Intel Data Center GPU Max

3. **CUDA VMM Support**: Use CUDA Virtual Memory Management for more flexible memory handling

4. **Multi-GPU Support**: Current sysfs-based topology mapping covers PCIe domain/bus and NUMA affinity. Future work: NVML `nvmlDeviceGetTopologyCommonAncestor` for finer-grained PCIe switch distance

5. **Accelerator Memory Pool**: Pre-allocated buffer pools for reduced allocation overhead

6. **NCCL Integration**: Direct integration with NCCL for distributed training scenarios

## References

- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Linux DMA-BUF](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html)
- [RDMA Core Programming](https://www.rdmamojo.com/)
