#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/net/ib/GpuMemory.h"
#include "common/utils/Result.h"

namespace hf3fs::net {

/**
 * GPU Memory Import Support
 *
 * This module provides mechanisms for importing GPU memory from external
 * processes and registering it for RDMA operations. This is essential for
 * scenarios where:
 *
 * 1. The usrbio client runs in the inference engine process
 * 2. GPU memory is allocated with the inference engine's CUDA context
 * 3. The fuse daemon needs to register this memory for RDMA to storage
 *
 * The key challenge is CUDA context ownership: GPU memory is associated with
 * a specific CUDA context, and operations on that memory must be performed
 * from within that context. However, for GDR to work, the RDMA driver needs
 * to be able to access the GPU memory from the fuse daemon process.
 *
 * Solutions supported:
 *
 * 1. CUDA IPC (Inter-Process Communication)
 *    - cudaIpcGetMemHandle() exports memory from owner process
 *    - cudaIpcOpenMemHandle() imports memory in consumer process
 *    - Consumer gets a device pointer valid in their CUDA context
 *    - Memory can then be registered with RDMA
 *
 * 2. DMA-BUF Export/Import (Linux kernel 5.12+)
 *    - CUDA exports GPU memory as dmabuf fd
 *    - fd can be passed to another process via Unix socket
 *    - ibv_reg_dmabuf_mr() registers dmabuf with RDMA
 *    - Does NOT require CUDA context in the consumer
 *
 * 3. Nvidia peermem (BAR1 mapping)
 *    - nvidia_peermem kernel module
 *    - Direct mapping of GPU memory for peer access
 *    - Requires same machine (no cross-node)
 *
 * The dmabuf approach is preferred when available as it:
 * - Doesn't require CUDA in the fuse daemon
 * - Properly handles memory lifetime
 * - Works with any process that has the fd
 */

/**
 * Import method to use for GPU memory
 */
enum class GpuImportMethod {
  Auto,       // Automatically choose best method
  CudaIpc,    // CUDA IPC handles
  DmaBuf,     // Linux dmabuf
  DirectReg,  // Direct registration (nvidia_peermem)
};

/**
 * Configuration for GPU memory import
 */
class GpuImportConfig : public ConfigBase<GpuImportConfig> {
 public:
  CONFIG_ITEM(method, GpuImportMethod::Auto);
  CONFIG_ITEM(enable_peer_access, true);
  CONFIG_ITEM(cache_imported_regions, true);
  CONFIG_ITEM(verify_import_success, true);
};

/**
 * Exported GPU memory handle
 *
 * This structure contains all information needed to import GPU memory
 * in another process. It can be serialized and sent via IPC.
 */
struct GpuExportHandle {
  // CUDA IPC handle (64 bytes)
  uint8_t ipcHandle[64];
  bool hasIpcHandle = false;

  // dmabuf file descriptor
  // Note: fd must be sent via Unix socket SCM_RIGHTS
  int dmabufFd = -1;
  bool hasDmabuf = false;

  // Memory info
  uint64_t devicePtrValue = 0;  // Original device pointer (as integer)
  size_t size = 0;
  int deviceId = -1;
  size_t alignment = 0;

  // Serialization (excludes fd which must be sent separately)
  std::string serialize() const;
  static Result<GpuExportHandle> deserialize(const std::string& data);
};

/**
 * Imported GPU memory region
 *
 * Represents GPU memory that has been imported from another process
 * and registered with the RDMA subsystem.
 */
class GpuImportedRegion {
 public:
  ~GpuImportedRegion();

  // Non-copyable
  GpuImportedRegion(const GpuImportedRegion&) = delete;
  GpuImportedRegion& operator=(const GpuImportedRegion&) = delete;

  // Movable
  GpuImportedRegion(GpuImportedRegion&&) noexcept;
  GpuImportedRegion& operator=(GpuImportedRegion&&) noexcept;

  /**
   * Create by importing from export handle
   *
   * @param handle Export handle from the owning process
   * @param config Import configuration
   * @return Imported region or error
   */
  static Result<std::unique_ptr<GpuImportedRegion>> import(
      const GpuExportHandle& handle,
      const GpuImportConfig& config = GpuImportConfig());

  /**
   * Create by importing dmabuf fd directly
   *
   * This method doesn't require CUDA at all in the importing process.
   *
   * @param dmabufFd File descriptor for dmabuf
   * @param size Size of the memory region
   * @param deviceId GPU device ID (for informational purposes)
   * @param config Import configuration
   * @return Imported region or error
   */
  static Result<std::unique_ptr<GpuImportedRegion>> importDmabuf(
      int dmabufFd,
      size_t size,
      int deviceId,
      const GpuImportConfig& config = GpuImportConfig());

  // Accessors
  void* ptr() const { return importedPtr_; }
  size_t size() const { return size_; }
  int deviceId() const { return deviceId_; }
  GpuImportMethod method() const { return method_; }

  /**
   * Get the underlying GPU memory region for RDMA operations
   */
  std::shared_ptr<GpuMemoryRegion> getRegion() const { return region_; }

  /**
   * Get memory region for specific IB device
   */
  ibv_mr* getMR(int devId) const {
    return region_ ? region_->getMR(devId) : nullptr;
  }

  /**
   * Get rkey for specific IB device
   */
  std::optional<uint32_t> getRkey(int devId) const {
    return region_ ? region_->getRkey(devId) : std::nullopt;
  }

 private:
  GpuImportedRegion() = default;

  Result<Void> doImport(const GpuExportHandle& handle, const GpuImportConfig& config);
  Result<Void> doImportDmabuf(int dmabufFd, const GpuImportConfig& config);
  void cleanup();

  void* importedPtr_ = nullptr;
  size_t size_ = 0;
  int deviceId_ = -1;
  GpuImportMethod method_ = GpuImportMethod::Auto;

  // Resources to cleanup
  bool ownsIpcHandle_ = false;
  int ownedDmabufFd_ = -1;

  // RDMA region
  std::shared_ptr<GpuMemoryRegion> region_;
};

/**
 * GPU Memory Exporter
 *
 * Used by the process that owns the GPU memory to create export handles.
 */
class GpuMemoryExporter {
 public:
  /**
   * Export GPU memory for sharing with other processes
   *
   * @param devicePtr GPU device pointer
   * @param size Size of the memory region
   * @param deviceId CUDA device ID
   * @param method Preferred export method
   * @return Export handle or error
   */
  static Result<GpuExportHandle> exportMemory(
      void* devicePtr,
      size_t size,
      int deviceId,
      GpuImportMethod method = GpuImportMethod::Auto);

  /**
   * Check if dmabuf export is available
   */
  static bool isDmabufSupported();

  /**
   * Check if CUDA IPC is available
   */
  static bool isCudaIpcSupported();
};

/**
 * GPU Memory Import Manager
 *
 * Manages imported GPU memory regions with caching for efficiency.
 */
class GpuImportManager {
 public:
  static GpuImportManager& instance();

  /**
   * Import GPU memory using the provided handle
   *
   * @param handle Export handle
   * @param config Import configuration
   * @return Imported region
   */
  Result<std::shared_ptr<GpuImportedRegion>> import(
      const GpuExportHandle& handle,
      const GpuImportConfig& config = GpuImportConfig());

  /**
   * Import GPU memory using dmabuf fd
   *
   * @param dmabufFd dmabuf file descriptor
   * @param size Memory size
   * @param deviceId GPU device ID
   * @param config Import configuration
   * @return Imported region
   */
  Result<std::shared_ptr<GpuImportedRegion>> importDmabuf(
      int dmabufFd,
      size_t size,
      int deviceId,
      const GpuImportConfig& config = GpuImportConfig());

  /**
   * Invalidate a cached import by device pointer
   */
  void invalidate(uint64_t devicePtrValue);

  /**
   * Clear all cached imports
   */
  void clear();

  /**
   * Get statistics
   */
  struct Stats {
    size_t activeImports = 0;
    size_t totalImported = 0;
    size_t cacheHits = 0;
    size_t cacheMisses = 0;
  };
  Stats getStats() const;

 private:
  GpuImportManager() = default;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::weak_ptr<GpuImportedRegion>> cache_;
  Stats stats_;
};

/**
 * Helper function to send dmabuf fd via Unix socket
 *
 * @param sockFd Unix socket file descriptor
 * @param dmabufFd dmabuf file descriptor to send
 * @return true on success
 */
bool sendDmabufFd(int sockFd, int dmabufFd);

/**
 * Helper function to receive dmabuf fd via Unix socket
 *
 * @param sockFd Unix socket file descriptor
 * @return Received fd or -1 on error
 */
int recvDmabufFd(int sockFd);

}  // namespace hf3fs::net
