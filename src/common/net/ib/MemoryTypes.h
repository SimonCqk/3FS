#pragma once

namespace hf3fs::net {

/**
 * Memory type enumeration (vendor-neutral)
 * Following industry conventions (UCX, libfabric)
 */
enum class MemoryType {
    Host = 0,           // System/CPU memory
    Device,             // Accelerator device memory
    Managed,            // Unified/managed memory
    Pinned,             // Host memory pinned for DMA
    Unknown
};

/**
 * Vendor identification (for dispatch)
 */
enum class DeviceVendor {
    None = 0,           // Host memory
    NVIDIA,             // CUDA
    AMD,                // ROCm/HIP
    Intel,              // Level Zero
};

}  // namespace hf3fs::net
