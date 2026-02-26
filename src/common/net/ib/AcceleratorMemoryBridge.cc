#include "AcceleratorMemoryBridge.h"

#include <cstring>
#include <unistd.h>

#include <folly/logging/xlog.h>
#include <fmt/format.h>

#ifdef HF3FS_GDR_ENABLED
#include <cuda_runtime.h>
#endif

namespace hf3fs::net {

// AcceleratorExportHandle implementation

std::string AcceleratorExportHandle::serialize() const {
  // Format: [1 byte flags][64 bytes ipc][8 bytes ptr][8 bytes size][4 bytes deviceId][8 bytes alignment]
  std::string result(1 + 64 + 8 + 8 + 4 + 8, '\0');

  uint8_t flags = 0;
  if (hasIpcHandle) flags |= 0x01;

  size_t offset = 0;
  result[offset++] = flags;
  std::memcpy(&result[offset], ipcHandle, 64); offset += 64;
  std::memcpy(&result[offset], &devicePtrValue, 8); offset += 8;
  std::memcpy(&result[offset], &size, 8); offset += 8;
  std::memcpy(&result[offset], &deviceId, 4); offset += 4;
  std::memcpy(&result[offset], &alignment, 8); offset += 8;

  return result;
}

Result<AcceleratorExportHandle> AcceleratorExportHandle::deserialize(const std::string& data) {
   if (data.size() != 1 + 64 + 8 + 8 + 4 + 8) {
     return makeError(StatusCode::kInvalidArg, "Invalid export handle data size");
   }

   AcceleratorExportHandle handle;
  size_t offset = 0;

  uint8_t flags = data[offset++];
  handle.hasIpcHandle = (flags & 0x01) != 0;

  std::memcpy(handle.ipcHandle, &data[offset], 64); offset += 64;
  std::memcpy(&handle.devicePtrValue, &data[offset], 8); offset += 8;
  std::memcpy(&handle.size, &data[offset], 8); offset += 8;
  std::memcpy(&handle.deviceId, &data[offset], 4); offset += 4;
  std::memcpy(&handle.alignment, &data[offset], 8); offset += 8;

  return handle;
}

// AcceleratorImportedRegion implementation

AcceleratorImportedRegion::~AcceleratorImportedRegion() {
  cleanup();
}

AcceleratorImportedRegion::AcceleratorImportedRegion(AcceleratorImportedRegion&& other) noexcept
    : importedPtr_(other.importedPtr_),
      size_(other.size_),
      deviceId_(other.deviceId_),
      method_(other.method_),
      ownsIpcHandle_(other.ownsIpcHandle_),
      region_(std::move(other.region_)) {
  other.importedPtr_ = nullptr;
  other.ownsIpcHandle_ = false;
}

AcceleratorImportedRegion& AcceleratorImportedRegion::operator=(AcceleratorImportedRegion&& other) noexcept {
  if (this != &other) {
    cleanup();

    importedPtr_ = other.importedPtr_;
    size_ = other.size_;
    deviceId_ = other.deviceId_;
    method_ = other.method_;
    ownsIpcHandle_ = other.ownsIpcHandle_;
    region_ = std::move(other.region_);

    other.importedPtr_ = nullptr;
    other.ownsIpcHandle_ = false;
  }
  return *this;
}

Result<std::unique_ptr<AcceleratorImportedRegion>> AcceleratorImportedRegion::import(
     const AcceleratorExportHandle& handle,
     const AcceleratorImportConfig& config) {
   auto region = std::unique_ptr<AcceleratorImportedRegion>(new AcceleratorImportedRegion());

  auto result = region->doImport(handle, config);
  if (!result) {
    return makeError(result.error());
  }

  return std::move(region);
}

Result<Void> AcceleratorImportedRegion::doImport(
     const AcceleratorExportHandle& handle,
     const AcceleratorImportConfig& config) {
  size_ = handle.size;
  deviceId_ = handle.deviceId;

   // Determine best import method
   AcceleratorImportMethod method = config.method();
   if (method == AcceleratorImportMethod::Auto) {
     if (handle.hasIpcHandle) {
       method = AcceleratorImportMethod::CudaIpc;
     } else {
       method = AcceleratorImportMethod::DirectReg;
     }
   }

   method_ = method;

   switch (method) {
     case AcceleratorImportMethod::CudaIpc:
      if (!handle.hasIpcHandle) {
        return makeError(StatusCode::kInvalidArg, "IPC handle not available");
      }
#ifdef HF3FS_GDR_ENABLED
      {
        cudaError_t err = cudaSetDevice(deviceId_);
        if (err != cudaSuccess) {
          return makeError(StatusCode::kIOError,
                           fmt::format("cudaSetDevice({}) failed: {}",
                                       deviceId_, cudaGetErrorString(err)));
        }
        cudaIpcMemHandle_t cudaHandle;
        std::memcpy(&cudaHandle, handle.ipcHandle, sizeof(cudaHandle));
        err = cudaIpcOpenMemHandle(&importedPtr_, cudaHandle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
          return makeError(StatusCode::kIOError,
                           fmt::format("cudaIpcOpenMemHandle failed: {}",
                                       cudaGetErrorString(err)));
        }
        ownsIpcHandle_ = true;
      }
      break;
#else
      return makeError(StatusCode::kNotImplemented, "CUDA IPC not supported in this build");
#endif

     case AcceleratorImportMethod::DirectReg:
       // Direct registration using the original pointer
       // This only works if nvidia_peermem is loaded and the process has access
       importedPtr_ = reinterpret_cast<void*>(handle.devicePtrValue);
       break;

     default:
       return makeError(StatusCode::kInvalidArg, "Invalid import method");
  }

  // Create GPU memory region for RDMA
  AcceleratorMemoryDescriptor desc;
  desc.devicePtr = importedPtr_;
  desc.size = size_;
  desc.deviceId = deviceId_;
  if (handle.hasIpcHandle) {
    std::memcpy(desc.ipcHandle.data, handle.ipcHandle, sizeof(desc.ipcHandle.data));
    desc.ipcHandle.valid = true;
  }

  if (GDRManager::instance().isAvailable()) {
    auto regionResult = AcceleratorMemoryRegion::create(desc, GDRManager::instance().config());
    if (!regionResult) {
      XLOGF(ERR, "Failed to create GPU memory region: {}", regionResult.error().message());
      cleanup();
      return makeError(regionResult.error());
    }
    region_ = std::move(*regionResult);
  }

  XLOGF(INFO, "GPU memory imported: method={}, ptr={}, size={}, device={}",
        static_cast<int>(method_), importedPtr_, size_, deviceId_);

  return Void{};
}

void AcceleratorImportedRegion::cleanup() {
  region_.reset();

  if (ownsIpcHandle_ && importedPtr_) {
#ifdef HF3FS_GDR_ENABLED
    cudaError_t err = cudaIpcCloseMemHandle(importedPtr_);
    if (err != cudaSuccess) {
      XLOGF(WARN, "cudaIpcCloseMemHandle failed: {}", cudaGetErrorString(err));
    }
#else
    XLOGF(DBG, "Closing CUDA IPC handle");
#endif
  }

  importedPtr_ = nullptr;
  ownsIpcHandle_ = false;
}

// AcceleratorMemoryExporter implementation

Result<AcceleratorExportHandle> AcceleratorMemoryExporter::exportMemory(
     void* devicePtr,
     size_t size,
     int deviceId,
     AcceleratorImportMethod method) {
  if (!devicePtr || size == 0) {
    return makeError(StatusCode::kInvalidArg, "Invalid memory parameters");
  }

   AcceleratorExportHandle handle;
   handle.devicePtrValue = reinterpret_cast<uint64_t>(devicePtr);
   handle.size = size;
   handle.deviceId = deviceId;

   // Determine export method
   if (method == AcceleratorImportMethod::Auto) {
     if (isCudaIpcSupported()) {
       method = AcceleratorImportMethod::CudaIpc;
     } else {
       method = AcceleratorImportMethod::DirectReg;
     }
   }

   switch (method) {
     case AcceleratorImportMethod::CudaIpc:
       // Export as CUDA IPC handle
#ifdef HF3FS_GDR_ENABLED
      {
        cudaIpcMemHandle_t cudaHandle;
        cudaError_t err = cudaIpcGetMemHandle(&cudaHandle, devicePtr);
        if (err != cudaSuccess) {
          return makeError(StatusCode::kIOError,
                           fmt::format("cudaIpcGetMemHandle failed: {}",
                                       cudaGetErrorString(err)));
        }
        std::memcpy(handle.ipcHandle, &cudaHandle, sizeof(handle.ipcHandle));
        handle.hasIpcHandle = true;
      }
      break;
#else
      XLOGF(WARN, "CUDA IPC export requires CUDA runtime - not supported in this build");
      handle.hasIpcHandle = false;
      break;
#endif

     case AcceleratorImportMethod::DirectReg:
       // Direct registration doesn't need export - just pass the pointer
       XLOGF(DBG, "Using direct registration method for GPU memory");
       break;

     default:
       return makeError(StatusCode::kInvalidArg, "Invalid export method");
  }

  XLOGF(INFO, "GPU memory exported: ptr={}, size={}, device={}, ipc={}",
        devicePtr, size, deviceId, handle.hasIpcHandle);

  return handle;
}

bool AcceleratorMemoryExporter::isCudaIpcSupported() {
#ifdef HF3FS_GDR_ENABLED
  return true;
#else
  return false;
#endif
}

// AcceleratorImportManager implementation

AcceleratorImportManager& AcceleratorImportManager::instance() {
   static AcceleratorImportManager instance;
   return instance;
}

Result<std::shared_ptr<AcceleratorImportedRegion>> AcceleratorImportManager::import(
     const AcceleratorExportHandle& handle,
     const AcceleratorImportConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check cache
  auto it = cache_.find(handle.devicePtrValue);
  if (it != cache_.end()) {
    auto region = it->second.lock();
    if (region) {
      ++stats_.cacheHits;
      return region;
    }
    cache_.erase(it);
  }

  ++stats_.cacheMisses;

   // Create new import
   auto result = AcceleratorImportedRegion::import(handle, config);
   if (!result) {
     return makeError(result.error());
   }

   auto region = std::shared_ptr<AcceleratorImportedRegion>(std::move(*result));

  // Cache if configured
  if (config.cache_imported_regions()) {
    cache_[handle.devicePtrValue] = region;
  }

  ++stats_.totalImported;
  ++stats_.activeImports;

  return region;
}

void AcceleratorImportManager::invalidate(uint64_t devicePtrValue) {
   std::lock_guard<std::mutex> lock(mutex_);
   cache_.erase(devicePtrValue);
}

void AcceleratorImportManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

AcceleratorImportManager::Stats AcceleratorImportManager::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto stats = stats_;

  // Count active imports
  stats.activeImports = 0;
  for (const auto& [key, weakRegion] : cache_) {
    if (!weakRegion.expired()) {
      ++stats.activeImports;
    }
  }

  return stats;
}

}  // namespace hf3fs::net
