/**
 * GPU Direct RDMA (GDR) Extension Implementation
 *
 * This implements the simplified GDR API that mirrors the standard usrbio
 * interface. All CUDA complexity is hidden internally.
 */

#include "hf3fs_usrbio_gdr.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <folly/logging/xlog.h>
#include <fmt/format.h>

#ifdef HF3FS_GDR_ENABLED
#include <cuda_runtime.h>
#endif

#include "common/logging/LogInit.h"
#include "common/net/ib/GpuMemory.h"
#include "common/net/ib/IBDevice.h"
#include "common/utils/Uuid.h"

namespace {

// Magic value to identify GPU iovs (stored in numa field)
constexpr int kGpuIovMagicNuma = -0x6472;  // "gdr" in hex-ish

/**
 * Internal handle for GPU iov management
 *
 * This structure contains all GPU-specific state, hidden from the user.
 * It is pointed to by iov->iovh for GPU memory iovs.
 */
struct GpuIovHandle {
  // GPU memory region registered with RDMA
  std::shared_ptr<hf3fs::net::GpuMemoryRegion> region;

  // GPU device ID
  int deviceId = -1;

  // Original device pointer
  void* devicePtr = nullptr;

  // Whether memory was allocated by this library (needs cudaFree on destroy)
  bool ownsMemory = false;

  // Whether this was imported via IPC
  bool isIpcImported = false;

  // IPC handle for cross-process sharing
  hf3fs::net::GpuMemoryDescriptor::IpcHandle ipcHandle;

  // Memory size
  size_t size = 0;

  ~GpuIovHandle() {
    // Free GPU memory if we own it
    if (ownsMemory && devicePtr) {
      XLOGF(DBG, "Freeing GPU memory: ptr={}, size={}, device={}",
            static_cast<const void*>(devicePtr), size, deviceId);
#ifdef HF3FS_GDR_ENABLED
      cudaError_t err = cudaSetDevice(deviceId);
      if (err != cudaSuccess) {
        XLOGF(WARN, "cudaSetDevice({}) failed: {}", deviceId, cudaGetErrorString(err));
      } else {
        err = cudaFree(devicePtr);
        if (err != cudaSuccess) {
          XLOGF(WARN, "cudaFree failed: {}", cudaGetErrorString(err));
        }
      }
#endif
    }

    // Close IPC handle if imported
    if (isIpcImported && devicePtr) {
      XLOGF(DBG, "Closing IPC imported handle: ptr={}",
            static_cast<const void*>(devicePtr));
#ifdef HF3FS_GDR_ENABLED
      cudaError_t err = cudaSetDevice(deviceId);
      if (err != cudaSuccess) {
        XLOGF(WARN, "cudaSetDevice({}) failed: {}", deviceId, cudaGetErrorString(err));
      }
      err = cudaIpcCloseMemHandle(devicePtr);
      if (err != cudaSuccess) {
        XLOGF(WARN, "cudaIpcCloseMemHandle failed: {}", cudaGetErrorString(err));
      }
#endif
    }

    // Region cleanup is automatic via shared_ptr
  }
};

// Global registry of GPU iov handles (for tracking and cleanup)
std::mutex gGpuIovMutex;
std::unordered_map<void*, std::unique_ptr<GpuIovHandle>> gGpuIovHandles;

// GDR initialization state
std::once_flag gGdrInitOnce;
bool gGdrInitialized = false;

/**
 * Ensure GDR is initialized (lazy initialization)
 */
bool ensureGdrInitialized() {
  std::call_once(gGdrInitOnce, []() {
    // Initialize logging
    auto logLevel = getenv("HF3FS_USRBIO_LIB_LOG");
    hf3fs::logging::initOrDie(logLevel && *logLevel ? logLevel : "WARN");

    XLOGF(INFO, "Initializing GDR support");

    // Check IB availability
    if (!hf3fs::net::IBManager::initialized()) {
      XLOGF(WARN, "IB not initialized - GDR may have limited functionality");
    }

    // Initialize GDR manager
    hf3fs::net::GDRConfig config;
    config.set_enabled(true);

    auto result = hf3fs::net::GDRManager::instance().init(config);
    if (!result) {
      XLOGF(ERR, "GDR initialization failed: {}", result.error().message());
      return;
    }

    gGdrInitialized = true;
    XLOGF(INFO, "GDR support initialized successfully");
  });

  return gGdrInitialized;
}

/**
 * Allocate GPU memory internally
 *
 * Encapsulates all CUDA memory allocation logic.
 *
 * @param size Bytes to allocate
 * @param deviceId CUDA device ID
 * @param[out] devicePtr Output device pointer
 * @return 0 on success, -errno on error
 */
int allocateGpuMemory(size_t size, int deviceId, void** devicePtr) {
  if (!devicePtr || size == 0) {
    return -EINVAL;
  }

  XLOGF(DBG, "Allocating GPU memory: size={}, device={}", size, deviceId);

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(deviceId);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaSetDevice({}) failed: {}", deviceId, cudaGetErrorString(err));
    return -ENODEV;
  }

  err = cudaMalloc(devicePtr, size);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaMalloc({}) failed: {}", size, cudaGetErrorString(err));
    return -ENOMEM;
  }

  XLOGF(INFO, "Allocated GPU memory: ptr={}, size={}, device={}",
        static_cast<const void*>(*devicePtr), size, deviceId);
  return 0;
#else
  XLOGF(WARN, "GPU memory allocation requires CUDA runtime - stub implementation");
  *devicePtr = nullptr;
  return -ENOTSUP;
#endif
}

/**
 * Free GPU memory
 *
 * Encapsulates CUDA memory deallocation.
 */
void freeGpuMemory(void* devicePtr, int deviceId) {
  if (!devicePtr) {
    return;
  }

  XLOGF(DBG, "Freeing GPU memory: ptr={}, device={}",
        static_cast<const void*>(devicePtr), deviceId);

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(deviceId);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaSetDevice({}) failed: {}", deviceId, cudaGetErrorString(err));
    return;
  }
  err = cudaFree(devicePtr);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaFree failed: {}", cudaGetErrorString(err));
  }
#endif
}

/**
 * Register a GPU iov handle
 */
void registerGpuIov(struct hf3fs_iov* iov, std::unique_ptr<GpuIovHandle> handle) {
  std::lock_guard<std::mutex> lock(gGpuIovMutex);
  gGpuIovHandles[iov->iovh] = std::move(handle);
}

/**
 * Unregister and cleanup a GPU iov handle
 */
std::unique_ptr<GpuIovHandle> unregisterGpuIov(struct hf3fs_iov* iov) {
  std::lock_guard<std::mutex> lock(gGpuIovMutex);
  auto it = gGpuIovHandles.find(iov->iovh);
  if (it != gGpuIovHandles.end()) {
    auto handle = std::move(it->second);
    gGpuIovHandles.erase(it);
    return handle;
  }
  return nullptr;
}

/**
 * Get GPU handle from iov (internal use)
 */
GpuIovHandle* getGpuHandle(const struct hf3fs_iov* iov) {
  if (!iov || iov->numa != kGpuIovMagicNuma || !iov->iovh) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(gGpuIovMutex);
  auto it = gGpuIovHandles.find(iov->iovh);
  return it != gGpuIovHandles.end() ? it->second.get() : nullptr;
}

/**
 * Create symlink for GPU iov registration with fuse daemon
 */
int createGpuIovSymlink(const struct hf3fs_iov* iov, int deviceId) {
  hf3fs::Uuid uuid;
  std::memcpy(uuid.data, iov->id, sizeof(uuid.data));

  auto link = fmt::format("{}/3fs-virt/iovs/{}.gdr.d{}",
                          iov->mount_point,
                          uuid.toHexString(),
                          deviceId);

  auto target = fmt::format("gdr://device/{}/ptr/{}/size/{}",
                            deviceId,
                            reinterpret_cast<uintptr_t>(iov->base),
                            iov->size);

  int result = symlink(target.c_str(), link.c_str());
  if (result < 0 && errno != EEXIST) {
    XLOGF(WARN, "Failed to create GDR symlink {} -> {}: {}",
          link, target, strerror(errno));
    return -errno;
  }

  XLOGF(DBG, "Created GDR symlink: {} -> {}", link, target);
  return 0;
}

/**
 * Remove symlink for GPU iov
 */
void removeGpuIovSymlink(const struct hf3fs_iov* iov, int deviceId) {
  hf3fs::Uuid uuid;
  std::memcpy(uuid.data, iov->id, sizeof(uuid.data));

  auto link = fmt::format("{}/3fs-virt/iovs/{}.gdr.d{}",
                          iov->mount_point,
                          uuid.toHexString(),
                          deviceId);

  unlink(link.c_str());
}

}  // namespace

extern "C" {

bool hf3fs_gdr_available(void) {
  if (!ensureGdrInitialized()) {
    return false;
  }
  auto& mgr = hf3fs::net::GDRManager::instance();
  // GDR is only available if initialized AND has a valid region cache
  return mgr.isAvailable() && mgr.getRegionCache() != nullptr;
}

int hf3fs_gdr_device_count(void) {
  if (!ensureGdrInitialized()) {
    return 0;
  }
  return static_cast<int>(hf3fs::net::GDRManager::instance().getGpuDevices().size());
}

int hf3fs_iovcreate_gpu(struct hf3fs_iov* iov,
                        const char* hf3fs_mount_point,
                        size_t size,
                        size_t block_size,
                        int gpu_device_id) {
  if (!iov || !hf3fs_mount_point || size == 0) {
    return -EINVAL;
  }

  if (!ensureGdrInitialized()) {
    return -ENOTSUP;
  }

  // Validate device ID
  int deviceCount = hf3fs_gdr_device_count();
  if (gpu_device_id < 0 || gpu_device_id >= deviceCount) {
    XLOGF(ERR, "Invalid GPU device ID: {} (have {} devices)",
          gpu_device_id, deviceCount);
    return -ENODEV;
  }

  // Validate mount point length
  if (strlen(hf3fs_mount_point) >= sizeof(iov->mount_point)) {
    XLOGF(ERR, "Mount point too long: {}", hf3fs_mount_point);
    return -EINVAL;
  }

  XLOGF(DBG, "Creating GPU iov: size={}, device={}", size, gpu_device_id);

  // Allocate GPU memory
  void* devicePtr = nullptr;
  int allocResult = allocateGpuMemory(size, gpu_device_id, &devicePtr);
  if (allocResult != 0) {
    return allocResult;
  }

  // Create internal handle
  auto handle = std::make_unique<GpuIovHandle>();
  handle->deviceId = gpu_device_id;
  handle->devicePtr = devicePtr;
  handle->ownsMemory = true;
  handle->size = size;

  // Create GPU memory descriptor for RDMA registration
  hf3fs::net::GpuMemoryDescriptor desc;
  desc.devicePtr = devicePtr;
  desc.size = size;
  desc.deviceId = gpu_device_id;

  // Register with RDMA subsystem
  auto* cache = hf3fs::net::GDRManager::instance().getRegionCache();
  if (!cache) {
    XLOGF(ERR, "GDR region cache not available");
    freeGpuMemory(devicePtr, gpu_device_id);
    return -ENOTSUP;
  }
  auto regionResult = cache->getOrCreate(desc);
  if (!regionResult) {
    XLOGF(ERR, "Failed to register GPU memory with RDMA: {}",
          regionResult.error().message());
    freeGpuMemory(devicePtr, gpu_device_id);
    return -ENOMEM;
  }
  handle->region = *regionResult;

  // Generate UUID for this iov
  hf3fs::Uuid uuid = hf3fs::Uuid::random();

  // Initialize the iov structure
  std::memset(iov, 0, sizeof(*iov));
  iov->base = static_cast<uint8_t*>(devicePtr);
  iov->size = size;
  iov->block_size = block_size;
  iov->numa = kGpuIovMagicNuma;  // Magic value to identify GPU iovs
  iov->iovh = handle.get();  // Store handle pointer
  std::memcpy(iov->id, uuid.data, sizeof(iov->id));
  std::strncpy(iov->mount_point, hf3fs_mount_point, sizeof(iov->mount_point) - 1);

  // Register with fuse daemon
  createGpuIovSymlink(iov, gpu_device_id);

  // Track the handle
  registerGpuIov(iov, std::move(handle));

  XLOGF(INFO, "Created GPU iov: ptr={}, size={}, device={}",
        static_cast<const void*>(devicePtr), size, gpu_device_id);

  return 0;
}

int hf3fs_iovopen_gpu(struct hf3fs_iov* iov,
                      const uint8_t id[16],
                      const char* hf3fs_mount_point,
                      size_t size,
                      size_t block_size,
                      int gpu_device_id) {
  if (!iov || !id || !hf3fs_mount_point || size == 0) {
    return -EINVAL;
  }

  if (!ensureGdrInitialized()) {
    return -ENOTSUP;
  }

  // Validate device ID
  int deviceCount = hf3fs_gdr_device_count();
  if (gpu_device_id < 0 || gpu_device_id >= deviceCount) {
    XLOGF(ERR, "Invalid GPU device ID: {}", gpu_device_id);
    return -ENODEV;
  }

  // Validate mount point length
  if (strlen(hf3fs_mount_point) >= sizeof(iov->mount_point)) {
    return -EINVAL;
  }

  hf3fs::Uuid uuid;
  std::memcpy(uuid.data, id, sizeof(uuid.data));
  XLOGF(DBG, "Opening GPU iov: id={}, size={}, device={}",
        uuid.toHexString(), size, gpu_device_id);

  // Look up existing iov in fuse namespace
  auto link = fmt::format("{}/3fs-virt/iovs/{}.gdr.d{}",
                          hf3fs_mount_point,
                          uuid.toHexString(),
                          gpu_device_id);

  char target[512];
  ssize_t len = readlink(link.c_str(), target, sizeof(target) - 1);
  if (len < 0) {
    XLOGF(ERR, "GPU iov not found: {}", link);
    return -ENOENT;
  }
  target[len] = '\0';

  // Parse the target: gdr://device/{id}/ptr/{ptr}/size/{size}
  int parsedDevice;
  uintptr_t parsedPtr;
  size_t parsedSize;
  if (sscanf(target, "gdr://device/%d/ptr/%lu/size/%lu",
             &parsedDevice, &parsedPtr, &parsedSize) != 3) {
    XLOGF(ERR, "Invalid GPU iov target: {}", target);
    return -EINVAL;
  }

  // Verify consistency
  if (static_cast<size_t>(parsedDevice) != static_cast<size_t>(gpu_device_id)) {
    XLOGF(ERR, "GPU device mismatch: expected {}, got {}", gpu_device_id, parsedDevice);
    return -EINVAL;
  }

  // Create handle for the existing GPU memory
  auto handle = std::make_unique<GpuIovHandle>();
  handle->deviceId = gpu_device_id;
  handle->devicePtr = reinterpret_cast<void*>(parsedPtr);
  handle->ownsMemory = false;  // We don't own it, just opening
  handle->size = parsedSize;

  // Register with RDMA
  hf3fs::net::GpuMemoryDescriptor desc;
  desc.devicePtr = handle->devicePtr;
  desc.size = parsedSize;
  desc.deviceId = gpu_device_id;

  auto* cache = hf3fs::net::GDRManager::instance().getRegionCache();
  if (!cache) {
    XLOGF(ERR, "GDR region cache not available");
    return -ENOTSUP;
  }
  auto regionResult = cache->getOrCreate(desc);
  if (!regionResult) {
    XLOGF(ERR, "Failed to register opened GPU memory: {}",
          regionResult.error().message());
    return -ENOMEM;
  }
  handle->region = *regionResult;

  // Initialize iov
  std::memset(iov, 0, sizeof(*iov));
  iov->base = static_cast<uint8_t*>(handle->devicePtr);
  iov->size = parsedSize;
  iov->block_size = block_size;
  iov->numa = kGpuIovMagicNuma;
  iov->iovh = handle.get();
  std::memcpy(iov->id, id, sizeof(iov->id));
  std::strncpy(iov->mount_point, hf3fs_mount_point, sizeof(iov->mount_point) - 1);

  registerGpuIov(iov, std::move(handle));

  XLOGF(INFO, "Opened GPU iov: id={}, ptr={}, size={}",
        uuid.toHexString(), static_cast<const void*>(iov->base), iov->size);

  return 0;
}

int hf3fs_iovwrap_gpu(struct hf3fs_iov* iov,
                      void* gpu_ptr,
                      const uint8_t id[16],
                      const char* hf3fs_mount_point,
                      size_t size,
                      size_t block_size,
                      int gpu_device_id) {
  if (!iov || !gpu_ptr || !id || !hf3fs_mount_point || size == 0) {
    return -EINVAL;
  }

  if (!ensureGdrInitialized()) {
    return -ENOTSUP;
  }

  // Validate device ID
  int deviceCount = hf3fs_gdr_device_count();
  if (deviceCount > 0 && (gpu_device_id < 0 || gpu_device_id >= deviceCount)) {
    XLOGF(ERR, "Invalid GPU device ID: {}", gpu_device_id);
    return -ENODEV;
  }

  // Validate mount point
  if (strlen(hf3fs_mount_point) >= sizeof(iov->mount_point)) {
    return -EINVAL;
  }

  XLOGF(DBG, "Wrapping GPU memory: ptr={}, size={}, device={}",
        static_cast<const void*>(gpu_ptr), size, gpu_device_id);

  // Create handle (we don't own the memory)
  auto handle = std::make_unique<GpuIovHandle>();
  handle->deviceId = gpu_device_id;
  handle->devicePtr = gpu_ptr;
  handle->ownsMemory = false;
  handle->size = size;

  // Create GPU memory descriptor
  hf3fs::net::GpuMemoryDescriptor desc;
  desc.devicePtr = gpu_ptr;
  desc.size = size;
  desc.deviceId = gpu_device_id;

  // Register with RDMA
  auto* cache = hf3fs::net::GDRManager::instance().getRegionCache();
  if (!cache) {
    XLOGF(ERR, "GDR region cache not available");
    return -ENOTSUP;
  }
  auto regionResult = cache->getOrCreate(desc);
  if (!regionResult) {
    XLOGF(ERR, "Failed to register GPU memory with RDMA: {}",
          regionResult.error().message());
    return -ENOMEM;
  }
  handle->region = *regionResult;

  // Initialize iov structure
  std::memset(iov, 0, sizeof(*iov));
  iov->base = static_cast<uint8_t*>(gpu_ptr);
  iov->size = size;
  iov->block_size = block_size;
  iov->numa = kGpuIovMagicNuma;
  iov->iovh = handle.get();
  std::memcpy(iov->id, id, sizeof(iov->id));
  std::strncpy(iov->mount_point, hf3fs_mount_point, sizeof(iov->mount_point) - 1);

  // Register with fuse daemon
  createGpuIovSymlink(iov, gpu_device_id);

  // Track handle
  registerGpuIov(iov, std::move(handle));

  XLOGF(INFO, "Wrapped GPU memory: ptr={}, size={}, device={}",
        static_cast<const void*>(gpu_ptr), size, gpu_device_id);

  return 0;
}

void hf3fs_iovunlink_gpu(struct hf3fs_iov* iov) {
  if (!iov) {
    return;
  }

  GpuIovHandle* handle = getGpuHandle(iov);
  if (!handle) {
    XLOGF(WARN, "hf3fs_iovunlink_gpu called on non-GPU iov");
    return;
  }

  // Remove the symlink from fuse namespace
  removeGpuIovSymlink(iov, handle->deviceId);

  XLOGF(DBG, "Unlinked GPU iov: ptr={}, device={}",
        static_cast<const void*>(iov->base), handle->deviceId);
}

void hf3fs_iovdestroy_gpu(struct hf3fs_iov* iov) {
  if (!iov) {
    return;
  }

  // Unlink first
  hf3fs_iovunlink_gpu(iov);

  // Get and remove the handle
  auto handle = unregisterGpuIov(iov);
  if (!handle) {
    XLOGF(WARN, "hf3fs_iovdestroy_gpu: no handle found");
    return;
  }

  XLOGF(DBG, "Destroying GPU iov: ptr={}, size={}, device={}, ownsMemory={}",
        static_cast<const void*>(handle->devicePtr),
        handle->size,
        handle->deviceId,
        handle->ownsMemory);

  // Invalidate the MR cache entry to prevent stale rkey reuse
  if (handle->devicePtr) {
    auto* cache = hf3fs::net::GDRManager::instance().getRegionCache();
    if (cache) {
      cache->invalidate(handle->devicePtr);
    }
  }

  // Handle destruction (including potential memory free) happens in destructor
  handle.reset();

  // Clear the iov structure
  std::memset(iov, 0, sizeof(*iov));
}

bool hf3fs_iov_is_gpu(const struct hf3fs_iov* iov) {
  if (!iov) {
    return false;
  }
  return iov->numa == kGpuIovMagicNuma && getGpuHandle(iov) != nullptr;
}

int hf3fs_iov_gpu_device(const struct hf3fs_iov* iov) {
  GpuIovHandle* handle = getGpuHandle(iov);
  return handle ? handle->deviceId : -1;
}

int hf3fs_iovsync_gpu(const struct hf3fs_iov* iov, int direction) {
  if (!iov) {
    return -EINVAL;
  }

  GpuIovHandle* handle = getGpuHandle(iov);
  if (!handle) {
    return -EINVAL;
  }

  XLOGF(DBG, "GPU sync: ptr={}, size={}, device={}, direction={}",
        static_cast<const void*>(handle->devicePtr),
        handle->size,
        handle->deviceId,
        direction);

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(handle->deviceId);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaSetDevice({}) failed: {}", handle->deviceId, cudaGetErrorString(err));
    return -ENODEV;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaDeviceSynchronize failed: {}", cudaGetErrorString(err));
    return -EIO;
  }
#else
  (void)direction;
#endif

  // For most cases, nvidia_peermem handles coherency automatically
  return 0;
}

int hf3fs_iov_export_gpu(const struct hf3fs_iov* iov,
                         hf3fs_gpu_ipc_handle_t* handle) {
  if (!iov || !handle) {
    return -EINVAL;
  }

  GpuIovHandle* gpuHandle = getGpuHandle(iov);
  if (!gpuHandle) {
    return -EINVAL;
  }

  XLOGF(DBG, "Exporting GPU IPC handle: ptr={}, device={}",
        static_cast<const void*>(gpuHandle->devicePtr),
        gpuHandle->deviceId);

  // Check if we have a valid device pointer to export
  if (!gpuHandle->devicePtr) {
    XLOGF(ERR, "Cannot export IPC handle: no valid GPU pointer");
    return -EINVAL;
  }

  // Check if we already have a valid IPC handle (from previous export or import)
  if (gpuHandle->ipcHandle.valid) {
    std::memset(handle, 0, sizeof(*handle));
    std::memcpy(handle->data, gpuHandle->ipcHandle.data,
                sizeof(gpuHandle->ipcHandle.data));
    auto metadata = reinterpret_cast<uint64_t*>(handle->data + 64);
    metadata[0] = gpuHandle->size;
    metadata[1] = static_cast<uint64_t>(gpuHandle->deviceId);
    return 0;
  }

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(gpuHandle->deviceId);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaSetDevice({}) failed: {}", gpuHandle->deviceId, cudaGetErrorString(err));
    return -ENODEV;
  }

  cudaIpcMemHandle_t cudaHandle;
  err = cudaIpcGetMemHandle(&cudaHandle, gpuHandle->devicePtr);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaIpcGetMemHandle failed: {}", cudaGetErrorString(err));
    return -EINVAL;
  }

  std::memset(handle, 0, sizeof(*handle));
  std::memcpy(handle->data, &cudaHandle, sizeof(cudaHandle));
  auto metadata = reinterpret_cast<uint64_t*>(handle->data + 64);
  metadata[0] = gpuHandle->size;
  metadata[1] = static_cast<uint64_t>(gpuHandle->deviceId);
  std::memcpy(gpuHandle->ipcHandle.data, &cudaHandle, sizeof(cudaHandle));
  gpuHandle->ipcHandle.valid = true;
  return 0;
#else
  XLOGF(ERR, "CUDA IPC export requires CUDA runtime - not implemented");
  return -ENOTSUP;
#endif
}

int hf3fs_iov_import_gpu(struct hf3fs_iov* iov,
                         const hf3fs_gpu_ipc_handle_t* handle,
                         const char* hf3fs_mount_point) {
  if (!iov || !handle || !hf3fs_mount_point) {
    return -EINVAL;
  }

  if (!ensureGdrInitialized()) {
    return -ENOTSUP;
  }

  // Extract metadata
  auto metadata = reinterpret_cast<const uint64_t*>(handle->data + 64);
  size_t size = metadata[0];
  int deviceId = static_cast<int>(metadata[1]);

  if (size == 0) {
    XLOGF(ERR, "Invalid IPC handle: size is 0");
    return -EINVAL;
  }

  XLOGF(DBG, "Importing GPU IPC handle: size={}, device={}", size, deviceId);

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(deviceId);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaSetDevice({}) failed: {}", deviceId, cudaGetErrorString(err));
    return -ENODEV;
  }

  cudaIpcMemHandle_t cudaHandle;
  std::memcpy(&cudaHandle, handle->data, sizeof(cudaHandle));

  void* importedPtr = nullptr;
  err = cudaIpcOpenMemHandle(&importedPtr, cudaHandle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaIpcOpenMemHandle failed: {}", cudaGetErrorString(err));
    return -EINVAL;
  }

  // Create handle for the imported GPU memory
  auto gpuHandle = std::make_unique<GpuIovHandle>();
  gpuHandle->deviceId = deviceId;
  gpuHandle->devicePtr = importedPtr;
  gpuHandle->ownsMemory = false;
  gpuHandle->isIpcImported = true;
  gpuHandle->size = size;
  std::memcpy(gpuHandle->ipcHandle.data, &cudaHandle, sizeof(cudaHandle));
  gpuHandle->ipcHandle.valid = true;

  hf3fs::net::GpuMemoryDescriptor desc;
  desc.devicePtr = importedPtr;
  desc.size = size;
  desc.deviceId = deviceId;
  std::memcpy(desc.ipcHandle.data, &cudaHandle, sizeof(desc.ipcHandle.data));
  desc.ipcHandle.valid = true;

  auto* cache = hf3fs::net::GDRManager::instance().getRegionCache();
  if (!cache) {
    XLOGF(ERR, "GDR region cache not available");
    cudaIpcCloseMemHandle(importedPtr);
    return -ENOTSUP;
  }
  auto regionResult = cache->getOrCreate(desc);
  if (!regionResult) {
    XLOGF(ERR, "Failed to register imported GPU memory: {}",
          regionResult.error().message());
    cudaIpcCloseMemHandle(importedPtr);
    return -ENOMEM;
  }
  gpuHandle->region = *regionResult;

  std::memset(iov, 0, sizeof(*iov));
  iov->base = static_cast<uint8_t*>(importedPtr);
  iov->size = size;
  iov->block_size = size;
  iov->numa = kGpuIovMagicNuma;
  iov->iovh = gpuHandle.get();
  std::memcpy(iov->mount_point, hf3fs_mount_point, sizeof(iov->mount_point) - 1);

  registerGpuIov(iov, std::move(gpuHandle));
  return 0;
#else
  XLOGF(ERR, "CUDA IPC import requires CUDA runtime - not implemented");
  return -ENOTSUP;
#endif
}

}  // extern "C"
