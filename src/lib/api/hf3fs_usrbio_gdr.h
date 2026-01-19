#pragma once

/**
 * hf3fs GPU Direct RDMA (GDR) Extension API
 *
 * This header provides extensions to the hf3fs usrbio API for enabling
 * GPU Direct RDMA, allowing data transfers directly between storage and
 * GPU memory without intermediate CPU copies.
 *
 * Design Goals:
 * - API consistency with standard usrbio interfaces (hf3fs_iovcreate, hf3fs_iovwrap, etc.)
 * - Hide CUDA complexity from users - no cudaMalloc calls required
 * - Minimal new types exposed - reuse hf3fs_iov where possible
 *
 * Usage (similar to standard usrbio):
 *
 *   // Create GPU iov (library allocates GPU memory internally)
 *   struct hf3fs_iov iov;
 *   hf3fs_iovcreate_gpu(&iov, "/mnt/hf3fs", size, 0, gpu_device_id);
 *
 *   // Or wrap existing GPU pointer
 *   hf3fs_iovwrap_gpu(&iov, existing_gpu_ptr, id, "/mnt/hf3fs", size, 0, gpu_device_id);
 *
 *   // Create ior and do I/O (same as regular usrbio)
 *   hf3fs_iorcreate4(&ior, "/mnt/hf3fs", entries, for_read, io_depth, timeout, -1, 0);
 *   hf3fs_prep_io(&ior, &iov, read, ptr, fd, off, len, userdata);
 *   hf3fs_submit_ios(&ior);
 *   hf3fs_wait_for_ios(&ior, cqes, cqec, min_results, timeout);
 *
 *   // Cleanup
 *   hf3fs_iovdestroy_gpu(&iov);
 *
 * Requirements:
 * - NVIDIA GPU with GPUDirect RDMA support
 * - nvidia_peermem kernel module loaded
 * - Mellanox/NVIDIA RDMA-capable NIC
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

#include "hf3fs_usrbio.h"

/**
 * Check if GDR support is available on this system
 *
 * This checks for:
 * - RDMA devices available
 * - nvidia_peermem module loaded
 * - At least one GDR-capable GPU
 *
 * @return true if GDR is available and can be used
 */
bool hf3fs_gdr_available(void);

/**
 * Get the number of GDR-capable GPU devices
 *
 * @return Number of GPU devices that support GDR, or 0 if none/error
 */
int hf3fs_gdr_device_count(void);

/**
 * Create a GPU memory iov with library-managed GPU memory allocation
 *
 * This function allocates GPU memory internally using CUDA and registers
 * it for RDMA operations. The user does not need to call any CUDA APIs.
 *
 * This is the GPU equivalent of hf3fs_iovcreate().
 *
 * @param iov Output parameter for the created iov (caller allocates struct)
 * @param hf3fs_mount_point Path to hf3fs mount point
 * @param size Size of GPU memory to allocate in bytes
 * @param block_size Block size hint (0 for default)
 * @param gpu_device_id CUDA device ID (0, 1, 2, ...) to allocate memory on
 * @return 0 on success, -errno on error
 *         -EINVAL if parameters are invalid
 *         -ENOMEM if GPU memory allocation fails
 *         -ENODEV if specified GPU device is not available
 *         -ENOTSUP if GDR is not supported
 */
int hf3fs_iovcreate_gpu(struct hf3fs_iov *iov,
                        const char *hf3fs_mount_point,
                        size_t size,
                        size_t block_size,
                        int gpu_device_id);

/**
 * Open an existing GPU memory iov by ID
 *
 * This is the GPU equivalent of hf3fs_iovopen().
 *
 * @param iov Output parameter for the opened iov
 * @param id UUID of the iov to open (16 bytes)
 * @param hf3fs_mount_point Path to hf3fs mount point
 * @param size Size of the GPU memory region
 * @param block_size Block size hint (0 for default)
 * @param gpu_device_id CUDA device ID
 * @return 0 on success, -errno on error
 */
int hf3fs_iovopen_gpu(struct hf3fs_iov *iov,
                      const uint8_t id[16],
                      const char *hf3fs_mount_point,
                      size_t size,
                      size_t block_size,
                      int gpu_device_id);

/**
 * Wrap existing GPU memory pointer as an iov
 *
 * Use this when you have GPU memory allocated externally (e.g., by a
 * deep learning framework) and want to use it for hf3fs I/O.
 *
 * This is the GPU equivalent of hf3fs_iovwrap().
 *
 * The GPU memory must remain valid for the lifetime of the iov.
 * The caller retains ownership of the GPU memory.
 *
 * @param iov Output parameter for the created iov (caller allocates struct)
 * @param gpu_ptr GPU device pointer (must be from cudaMalloc or similar)
 * @param id Unique identifier for this iov (16 bytes UUID)
 * @param hf3fs_mount_point Path to hf3fs mount point
 * @param size Size of the GPU memory region in bytes
 * @param block_size Block size hint (0 for default)
 * @param gpu_device_id CUDA device ID where the memory resides
 * @return 0 on success, -errno on error
 *         -EINVAL if parameters are invalid or gpu_ptr is not valid GPU memory
 *         -ENOMEM if RDMA registration fails
 */
int hf3fs_iovwrap_gpu(struct hf3fs_iov *iov,
                      void *gpu_ptr,
                      const uint8_t id[16],
                      const char *hf3fs_mount_point,
                      size_t size,
                      size_t block_size,
                      int gpu_device_id);

/**
 * Unlink a GPU memory iov (remove from hf3fs namespace)
 *
 * This is the GPU equivalent of hf3fs_iovunlink().
 *
 * @param iov The iov to unlink
 */
void hf3fs_iovunlink_gpu(struct hf3fs_iov *iov);

/**
 * Destroy a GPU memory iov and free associated resources
 *
 * For iovs created with hf3fs_iovcreate_gpu(), this also frees the GPU memory.
 * For iovs created with hf3fs_iovwrap_gpu(), only the RDMA registration is released.
 *
 * This is the GPU equivalent of hf3fs_iovdestroy().
 *
 * @param iov The iov to destroy (the struct itself is not freed)
 */
void hf3fs_iovdestroy_gpu(struct hf3fs_iov *iov);

/**
 * Check if an iov contains GPU memory
 *
 * @param iov The iov to check
 * @return true if this iov references GPU memory
 */
bool hf3fs_iov_is_gpu(const struct hf3fs_iov *iov);

/**
 * Get the GPU device ID for a GPU iov
 *
 * @param iov The iov to query
 * @return GPU device ID, or -1 if not a GPU iov
 */
int hf3fs_iov_gpu_device(const struct hf3fs_iov *iov);

/**
 * Synchronize GPU memory before/after RDMA operations
 *
 * In most cases, synchronization is handled automatically by the driver.
 * Use this function when you need explicit control over coherency, e.g.,
 * when interleaving GPU compute with storage I/O.
 *
 * @param iov The GPU iov to synchronize
 * @param direction 0 = before RDMA (ensure GPU writes visible to RDMA)
 *                  1 = after RDMA (ensure RDMA writes visible to GPU)
 * @return 0 on success, -errno on error
 */
int hf3fs_iovsync_gpu(const struct hf3fs_iov *iov, int direction);

/*
 * IPC Support for Cross-Process GPU Memory Sharing
 *
 * When the usrbio client runs in a different process than the fuse daemon
 * (typical in inference scenarios), use these functions to share GPU memory.
 *
 * Owner process (inference engine):
 *   1. Allocate GPU memory (via hf3fs_iovcreate_gpu or framework)
 *   2. Export IPC handle: hf3fs_iov_export_gpu()
 *   3. Send handle to fuse daemon via any IPC mechanism
 *
 * Consumer process (fuse daemon):
 *   1. Receive IPC handle
 *   2. Import: hf3fs_iov_import_gpu()
 *   3. Use for RDMA operations
 */

/**
 * GPU IPC handle for cross-process memory sharing
 *
 * This is an opaque handle that can be safely copied and transmitted
 * between processes (e.g., via shared memory, socket, pipe).
 */
typedef struct {
  uint8_t data[80];  // Opaque data containing CUDA IPC handle + metadata
} hf3fs_gpu_ipc_handle_t;

/**
 * Export a GPU iov for sharing with another process
 *
 * Creates an IPC handle that can be sent to another process.
 *
 * @param iov The GPU iov to export
 * @param handle Output parameter for the IPC handle
 * @return 0 on success, -errno on error
 *         -EINVAL if iov is not a GPU iov
 *         -ENOTSUP if IPC sharing is not supported
 */
int hf3fs_iov_export_gpu(const struct hf3fs_iov *iov,
                         hf3fs_gpu_ipc_handle_t *handle);

/**
 * Import a GPU memory iov from another process
 *
 * Creates a local iov that references GPU memory from another process.
 *
 * @param iov Output parameter for the imported iov
 * @param handle IPC handle from the exporting process
 * @param hf3fs_mount_point Path to hf3fs mount point
 * @return 0 on success, -errno on error
 */
int hf3fs_iov_import_gpu(struct hf3fs_iov *iov,
                         const hf3fs_gpu_ipc_handle_t *handle,
                         const char *hf3fs_mount_point);

#ifdef __cplusplus
}
#endif

/*
 * Note: Standard usrbio functions work transparently with GPU iovs:
 *
 * - hf3fs_prep_io(): Works with both regular and GPU iovs
 * - hf3fs_submit_ios(): No changes needed
 * - hf3fs_wait_for_ios(): No changes needed
 * - hf3fs_reg_fd() / hf3fs_dereg_fd(): No changes needed
 *
 * The library automatically detects GPU iovs and handles RDMA appropriately.
 */
