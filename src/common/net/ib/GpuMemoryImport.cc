#include "GpuMemoryImport.h"

#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

#include <folly/logging/xlog.h>
#include <fmt/format.h>

#ifdef HF3FS_GDR_ENABLED
#include <cuda_runtime.h>
#endif

namespace hf3fs::net {

// GpuExportHandle implementation

std::string GpuExportHandle::serialize() const {
  // Format: [1 byte flags][64 bytes ipc][8 bytes ptr][8 bytes size][4 bytes deviceId][8 bytes alignment]
  // Flags: bit 0 = hasIpcHandle, bit 1 = hasDmabuf
  std::string result(1 + 64 + 8 + 8 + 4 + 8, '\0');

  uint8_t flags = 0;
  if (hasIpcHandle) flags |= 0x01;
  if (hasDmabuf) flags |= 0x02;

  size_t offset = 0;
  result[offset++] = flags;
  std::memcpy(&result[offset], ipcHandle, 64); offset += 64;
  std::memcpy(&result[offset], &devicePtrValue, 8); offset += 8;
  std::memcpy(&result[offset], &size, 8); offset += 8;
  std::memcpy(&result[offset], &deviceId, 4); offset += 4;
  std::memcpy(&result[offset], &alignment, 8); offset += 8;

  return result;
}

Result<GpuExportHandle> GpuExportHandle::deserialize(const std::string& data) {
  if (data.size() != 1 + 64 + 8 + 8 + 4 + 8) {
    return makeError(StatusCode::kInvalidArg, "Invalid export handle data size");
  }

  GpuExportHandle handle;
  size_t offset = 0;

  uint8_t flags = data[offset++];
  handle.hasIpcHandle = (flags & 0x01) != 0;
  handle.hasDmabuf = (flags & 0x02) != 0;

  std::memcpy(handle.ipcHandle, &data[offset], 64); offset += 64;
  std::memcpy(&handle.devicePtrValue, &data[offset], 8); offset += 8;
  std::memcpy(&handle.size, &data[offset], 8); offset += 8;
  std::memcpy(&handle.deviceId, &data[offset], 4); offset += 4;
  std::memcpy(&handle.alignment, &data[offset], 8); offset += 8;

  return handle;
}

// GpuImportedRegion implementation

GpuImportedRegion::~GpuImportedRegion() {
  cleanup();
}

GpuImportedRegion::GpuImportedRegion(GpuImportedRegion&& other) noexcept
    : importedPtr_(other.importedPtr_),
      size_(other.size_),
      deviceId_(other.deviceId_),
      method_(other.method_),
      ownsIpcHandle_(other.ownsIpcHandle_),
      ownedDmabufFd_(other.ownedDmabufFd_),
      region_(std::move(other.region_)) {
  other.importedPtr_ = nullptr;
  other.ownsIpcHandle_ = false;
  other.ownedDmabufFd_ = -1;
}

GpuImportedRegion& GpuImportedRegion::operator=(GpuImportedRegion&& other) noexcept {
  if (this != &other) {
    cleanup();

    importedPtr_ = other.importedPtr_;
    size_ = other.size_;
    deviceId_ = other.deviceId_;
    method_ = other.method_;
    ownsIpcHandle_ = other.ownsIpcHandle_;
    ownedDmabufFd_ = other.ownedDmabufFd_;
    region_ = std::move(other.region_);

    other.importedPtr_ = nullptr;
    other.ownsIpcHandle_ = false;
    other.ownedDmabufFd_ = -1;
  }
  return *this;
}

Result<std::unique_ptr<GpuImportedRegion>> GpuImportedRegion::import(
    const GpuExportHandle& handle,
    const GpuImportConfig& config) {
  auto region = std::unique_ptr<GpuImportedRegion>(new GpuImportedRegion());

  auto result = region->doImport(handle, config);
  if (!result) {
    return makeError(result.error());
  }

  return std::move(region);
}

Result<std::unique_ptr<GpuImportedRegion>> GpuImportedRegion::importDmabuf(
    int dmabufFd,
    size_t size,
    int deviceId,
    const GpuImportConfig& config) {
  auto region = std::unique_ptr<GpuImportedRegion>(new GpuImportedRegion());

  region->size_ = size;
  region->deviceId_ = deviceId;

  auto result = region->doImportDmabuf(dmabufFd, config);
  if (!result) {
    return makeError(result.error());
  }

  return std::move(region);
}

Result<Void> GpuImportedRegion::doImport(
    const GpuExportHandle& handle,
    const GpuImportConfig& config) {
  size_ = handle.size;
  deviceId_ = handle.deviceId;

  // Determine best import method
  GpuImportMethod method = config.method();
  if (method == GpuImportMethod::Auto) {
    // Prefer dmabuf if available, then IPC
    if (handle.hasDmabuf && handle.dmabufFd >= 0) {
      method = GpuImportMethod::DmaBuf;
    } else if (handle.hasIpcHandle) {
      method = GpuImportMethod::CudaIpc;
    } else {
      // Try direct registration with the original pointer value
      method = GpuImportMethod::DirectReg;
    }
  }

  method_ = method;

  switch (method) {
    case GpuImportMethod::DmaBuf:
      if (!handle.hasDmabuf || handle.dmabufFd < 0) {
        return makeError(StatusCode::kInvalidArg, "dmabuf not available in handle");
      }
      return doImportDmabuf(handle.dmabufFd, config);

    case GpuImportMethod::CudaIpc:
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

    case GpuImportMethod::DirectReg:
      // Direct registration using the original pointer
      // This only works if nvidia_peermem is loaded and the process has access
      importedPtr_ = reinterpret_cast<void*>(handle.devicePtrValue);
      break;

    default:
      return makeError(StatusCode::kInvalidArg, "Invalid import method");
  }

  // Create GPU memory region for RDMA
  GpuMemoryDescriptor desc;
  desc.devicePtr = importedPtr_;
  desc.size = size_;
  desc.deviceId = deviceId_;
  desc.dmabufFd = (method == GpuImportMethod::DmaBuf) ? handle.dmabufFd : -1;
  if (handle.hasIpcHandle) {
    std::memcpy(desc.ipcHandle.data, handle.ipcHandle, sizeof(desc.ipcHandle.data));
    desc.ipcHandle.valid = true;
  }

  if (GDRManager::instance().isAvailable()) {
    auto regionResult = GpuMemoryRegion::create(desc, GDRManager::instance().config());
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

Result<Void> GpuImportedRegion::doImportDmabuf(
    int dmabufFd,
    const GpuImportConfig& config) {
  (void)config;
  if (dmabufFd < 0) {
    return makeError(StatusCode::kInvalidArg, "Invalid dmabuf fd");
  }

  method_ = GpuImportMethod::DmaBuf;

  // For dmabuf, we don't get a device pointer - we register the dmabuf directly with RDMA
  // The ibv_reg_dmabuf_mr() call handles all the mapping internally

  // Create GPU memory region from dmabuf
  if (GDRManager::instance().isAvailable()) {
    auto regionResult = GpuMemoryRegion::createFromDmabuf(
        dmabufFd, size_, deviceId_, GDRManager::instance().config());
    if (!regionResult) {
      XLOGF(ERR, "Failed to create GPU memory region from dmabuf: {}",
            regionResult.error().message());
      return makeError(regionResult.error());
    }
    region_ = std::move(*regionResult);
  }

  // Note: We don't get an importedPtr_ with dmabuf - the RDMA driver handles addressing internally
  // For GPU operations, the original process must still use the memory
  // This process only does RDMA to/from the memory

  XLOGF(INFO, "GPU memory imported via dmabuf: fd={}, size={}", dmabufFd, size_);

  return Void{};
}

void GpuImportedRegion::cleanup() {
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

  if (ownedDmabufFd_ >= 0) {
    close(ownedDmabufFd_);
  }

  importedPtr_ = nullptr;
  ownsIpcHandle_ = false;
  ownedDmabufFd_ = -1;
}

// GpuMemoryExporter implementation

Result<GpuExportHandle> GpuMemoryExporter::exportMemory(
    void* devicePtr,
    size_t size,
    int deviceId,
    GpuImportMethod method) {
  if (!devicePtr || size == 0) {
    return makeError(StatusCode::kInvalidArg, "Invalid memory parameters");
  }

  GpuExportHandle handle;
  handle.devicePtrValue = reinterpret_cast<uint64_t>(devicePtr);
  handle.size = size;
  handle.deviceId = deviceId;

  // Determine export method
  if (method == GpuImportMethod::Auto) {
    // Prefer dmabuf if available
    if (isDmabufSupported()) {
      method = GpuImportMethod::DmaBuf;
    } else if (isCudaIpcSupported()) {
      method = GpuImportMethod::CudaIpc;
    } else {
      method = GpuImportMethod::DirectReg;
    }
  }

  switch (method) {
    case GpuImportMethod::DmaBuf:
      // Export as dmabuf
      // In production with CUDA 11.2+:
      // CUmemGenericAllocationHandle allocHandle;
      // cuMemExportToShareableHandle(&shareHandle, allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
      // handle.dmabufFd = (int)shareHandle;
      XLOGF(WARN, "dmabuf export requires CUDA runtime with VMM support - placeholder");
      handle.hasDmabuf = false;
      // Fall through to try IPC
      [[fallthrough]];

    case GpuImportMethod::CudaIpc:
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

    case GpuImportMethod::DirectReg:
      // Direct registration doesn't need export - just pass the pointer
      XLOGF(DBG, "Using direct registration method for GPU memory");
      break;

    default:
      return makeError(StatusCode::kInvalidArg, "Invalid export method");
  }

  XLOGF(INFO, "GPU memory exported: ptr={}, size={}, device={}, ipc={}, dmabuf={}",
        devicePtr, size, deviceId, handle.hasIpcHandle, handle.hasDmabuf);

  return handle;
}

bool GpuMemoryExporter::isDmabufSupported() {
  // Check for dmabuf support:
  // 1. Kernel version >= 5.12 for ibv_reg_dmabuf_mr
  // 2. CUDA >= 11.2 for cuMemExportToShareableHandle with POSIX fd

#ifdef IBV_ACCESS_OPTIONAL_RANGE
  // IBV_ACCESS_OPTIONAL_RANGE is defined in newer RDMA headers that support dmabuf
  return true;
#else
  return false;
#endif
}

bool GpuMemoryExporter::isCudaIpcSupported() {
  // CUDA IPC requires:
  // 1. CUDA runtime available
  // 2. GPU supports IPC (most do, except some embedded)
  // 3. Same compute node (IPC is intra-node only)

#ifdef HF3FS_GDR_ENABLED
  return true;
#else
  return false;
#endif
}

// GpuImportManager implementation

GpuImportManager& GpuImportManager::instance() {
  static GpuImportManager instance;
  return instance;
}

Result<std::shared_ptr<GpuImportedRegion>> GpuImportManager::import(
    const GpuExportHandle& handle,
    const GpuImportConfig& config) {
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
  auto result = GpuImportedRegion::import(handle, config);
  if (!result) {
    return makeError(result.error());
  }

  auto region = std::shared_ptr<GpuImportedRegion>(std::move(*result));

  // Cache if configured
  if (config.cache_imported_regions()) {
    cache_[handle.devicePtrValue] = region;
  }

  ++stats_.totalImported;
  ++stats_.activeImports;

  return region;
}

Result<std::shared_ptr<GpuImportedRegion>> GpuImportManager::importDmabuf(
    int dmabufFd,
    size_t size,
    int deviceId,
    const GpuImportConfig& config) {
  // dmabuf imports are not cached by default (fd is not a stable key)
  auto result = GpuImportedRegion::importDmabuf(dmabufFd, size, deviceId, config);
  if (!result) {
    return makeError(result.error());
  }

  auto region = std::shared_ptr<GpuImportedRegion>(std::move(*result));

  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.totalImported;
    ++stats_.activeImports;
  }

  return region;
}

void GpuImportManager::invalidate(uint64_t devicePtrValue) {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.erase(devicePtrValue);
}

void GpuImportManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

GpuImportManager::Stats GpuImportManager::getStats() const {
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

// Helper functions for dmabuf fd passing

bool sendDmabufFd(int sockFd, int dmabufFd) {
  if (sockFd < 0 || dmabufFd < 0) {
    return false;
  }

  // Use SCM_RIGHTS to send the fd
  struct msghdr msg = {};
  struct iovec iov = {};
  char dummy = 'x';

  iov.iov_base = &dummy;
  iov.iov_len = 1;

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  // Control message with fd
  char control[CMSG_SPACE(sizeof(int))];
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &dmabufFd, sizeof(int));

  ssize_t sent = sendmsg(sockFd, &msg, 0);
  if (sent < 0) {
    XLOGF(ERR, "Failed to send dmabuf fd: {}", strerror(errno));
    return false;
  }

  return true;
}

int recvDmabufFd(int sockFd) {
  if (sockFd < 0) {
    return -1;
  }

  struct msghdr msg = {};
  struct iovec iov = {};
  char dummy;

  iov.iov_base = &dummy;
  iov.iov_len = 1;

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  char control[CMSG_SPACE(sizeof(int))];
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);

  ssize_t received = recvmsg(sockFd, &msg, 0);
  if (received < 0) {
    XLOGF(ERR, "Failed to receive dmabuf fd: {}", strerror(errno));
    return -1;
  }

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
    XLOGF(ERR, "Invalid control message for dmabuf fd");
    return -1;
  }

  int fd;
  std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));

  return fd;
}

}  // namespace hf3fs::net
