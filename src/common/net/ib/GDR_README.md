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

3. **Deprecated GPU API** (`hf3fs_usrbio_gdr.h`, `UsrbIoGdr.cc`)
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
   - CUDA IPC and dmabuf support
   - Handles CUDA context ownership issues

### Data Flow

```
Inference Engine Process              Fuse Daemon Process
┌─────────────────────────┐          ┌─────────────────────────┐
│  1. Create GPU IOV      │          │                         │
│     hf3fs_iovcreate_gpu │          │                         │
│                         │          │                         │
│  2. Export IPC Handle   │──────────▶ 3. Import IPC Handle   │
│     hf3fs_iov_export_gpu│   IPC    │    hf3fs_iov_import_gpu│
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
- Linux kernel 4.x+ (5.12+ recommended for dmabuf support)
- CUDA Toolkit 11.0+ (11.2+ for dmabuf export)
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

Use negative numa values to request accelerator device allocation:
- `numa >= 0`: Host memory on specified NUMA node
- `numa < 0`: Accelerator device, where `device_id = -(numa + 1)`
  - `numa = -1` → device 0
  - `numa = -2` → device 1
  - `numa = -3` → device 2

If GDR is unavailable, automatically falls back to host memory (numa = 0).

#### Unified API Example

```c
#include "lib/api/hf3fs_usrbio.h"

// Create IOV on GPU device 0 (or fallback to host if GDR unavailable)
struct hf3fs_iov iov;
int ret = hf3fs_iovcreate(&iov, "/mnt/hf3fs", size, 0, -1);  // -1 = device 0

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
hf3fs_iorcreate4(&ior, "/mnt/hf3fs", 64, true, 0, 0, -1, 0);
hf3fs_prep_io(&ior, &iov, true, iov.base, fd, 0, size, NULL);
hf3fs_submit_ios(&ior);
hf3fs_wait_for_ios(&ior, &cqe, 1, 1, NULL);

// Cleanup
hf3fs_iovdestroy(&iov);
hf3fs_iordestroy(&ior);
```

### Deprecated GPU API (Legacy)

The GPU-specific API in `hf3fs_usrbio_gdr.h` is deprecated but maintained for backward compatibility. All functions are marked with `__attribute__((deprecated))`.

#### Availability Check

```c
#include "lib/api/hf3fs_usrbio_gdr.h"

// Check if GDR is available on this system
if (hf3fs_gdr_available()) {
    int gpu_count = hf3fs_gdr_device_count();
    printf("GDR available with %d GPUs\n", gpu_count);
}
```

#### Creating GPU IOV (Library-Managed Memory) - Deprecated

```c
// DEPRECATED: Use hf3fs_iovcreate with negative numa instead
struct hf3fs_iov iov;
int ret = hf3fs_iovcreate_gpu(&iov, "/mnt/hf3fs", buffer_size, 0, gpu_device_id);
if (ret != 0) {
    fprintf(stderr, "Failed to create GPU iov: %d\n", ret);
    return ret;
}

// Create I/O ring (same as regular usrbio)
struct hf3fs_ior ior;
hf3fs_iorcreate4(&ior, "/mnt/hf3fs", 64, true, 0, 0, -1, 0);

// Register file
int fd = open("/mnt/hf3fs/data/file.bin", O_RDONLY);
hf3fs_reg_fd(fd, 0);

// Submit read to GPU memory (same as regular usrbio)
hf3fs_prep_io(&ior, &iov, true, iov.base, fd, 0, buffer_size, NULL);
hf3fs_submit_ios(&ior);

// Wait for completion
struct hf3fs_cqe cqe;
hf3fs_wait_for_ios(&ior, &cqe, 1, 1, NULL);

// Use data in GPU memory...

// Cleanup
hf3fs_iovdestroy_gpu(&iov);  // DEPRECATED: Use hf3fs_iovdestroy instead
hf3fs_iordestroy(&ior);
```

#### Wrapping Existing GPU Memory - Deprecated

Use this when you have GPU memory from an external source (e.g., deep learning framework):

```c
// DEPRECATED: Use hf3fs_iovwrap instead (auto-detects GPU memory)

// Existing GPU pointer from framework
void* gpu_ptr;  // e.g., from PyTorch, TensorFlow, or cudaMalloc

// Generate UUID for this iov
uint8_t uuid[16];
generate_uuid(uuid);

// Wrap the existing GPU memory
struct hf3fs_iov iov;
int ret = hf3fs_iovwrap_gpu(&iov, gpu_ptr, uuid, "/mnt/hf3fs",
                            buffer_size, 0, gpu_device_id);
if (ret != 0) {
    fprintf(stderr, "Failed to wrap GPU memory: %d\n", ret);
    return ret;
}

// Use for I/O (same as above)
hf3fs_prep_io(&ior, &iov, true, gpu_ptr, fd, offset, len, NULL);
hf3fs_submit_ios(&ior);

// Cleanup - only releases RDMA registration, does NOT free GPU memory
hf3fs_iovdestroy_gpu(&iov);  // DEPRECATED: Use hf3fs_iovdestroy instead
```

#### Cross-Process GPU Memory Sharing - Automatic

**Note:** IPC handles are now exported automatically when creating accelerator IOVs via the unified API. Manual export/import is no longer required for most use cases.

For scenarios where GPU memory is allocated in one process (inference engine) but RDMA operations happen in another (fuse daemon):

**Exporting Process (GPU memory owner):**
```c
// Create accelerator IOV - IPC handle exported automatically
struct hf3fs_iov iov;
hf3fs_iovcreate(&iov, "/mnt/hf3fs", size, 0, -1);  // -1 = device 0

// IPC handle is automatically exported and available for cross-process sharing
// No manual export needed
```

**Legacy Manual Export (Deprecated):**
```c
// DEPRECATED: IPC export is now automatic
struct hf3fs_iov iov;
hf3fs_iovcreate_gpu(&iov, "/mnt/hf3fs", size, 0, device_id);

// Export IPC handle for sharing
hf3fs_gpu_ipc_handle_t handle;
int ret = hf3fs_iov_export_gpu(&iov, &handle);
if (ret == 0) {
    // Send handle to consumer process via IPC mechanism
    send_to_consumer(handle);
}
```

**Importing Process (RDMA operator) - Deprecated:**
```c
// DEPRECATED: Use unified API instead
// Receive handle from owner process
hf3fs_gpu_ipc_handle_t handle;
recv_from_owner(&handle);

// Import the GPU memory
struct hf3fs_iov iov;
int ret = hf3fs_iov_import_gpu(&iov, &handle, "/mnt/hf3fs");
if (ret == 0) {
    // Can now use for RDMA operations
    hf3fs_prep_io(&ior, &iov, true, iov.base, fd, offset, len, NULL);
}

// Cleanup
hf3fs_iovdestroy_gpu(&iov);  // DEPRECATED: Use hf3fs_iovdestroy instead
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

4. **Multi-GPU Support**: Better topology-aware routing for multi-GPU systems

5. **Accelerator Memory Pool**: Pre-allocated buffer pools for reduced allocation overhead

6. **NCCL Integration**: Direct integration with NCCL for distributed training scenarios

## References

- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Linux DMA-BUF](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html)
- [RDMA Core Programming](https://www.rdmamojo.com/)
