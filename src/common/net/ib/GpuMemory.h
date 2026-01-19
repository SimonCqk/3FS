#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include <infiniband/verbs.h>

#include "common/net/ib/IBDevice.h"
#include "common/utils/ConfigBase.h"
#include "common/utils/Result.h"

namespace hf3fs::net {

/**
 * GPU Direct RDMA (GDR) Memory Management
 *
 * This module provides support for registering GPU memory with RDMA devices,
 * enabling direct data transfer between storage and GPU memory without CPU
 * involvement.
 *
 * Key features:
 * - GPU memory registration with IB devices
 * - Support for CUDA IPC memory handles for cross-process GPU memory sharing
 * - Memory region caching for efficient repeated access
 * - Support for memory import via IB dmabuf mechanism
 */

/**
 * Configuration for GDR functionality
 */
class GDRConfig : public ConfigBase<GDRConfig> {
 public:
  // Enable/disable GDR support globally
  CONFIG_ITEM(enabled, false);

  // Maximum number of GPU memory regions to cache
  CONFIG_ITEM(max_cached_regions, 1024u, ConfigCheckers::checkPositive);

  // Whether to use dmabuf for memory registration (preferred for cross-process)
  CONFIG_ITEM(use_dmabuf, true);

  // Timeout for memory registration operations (microseconds)
  CONFIG_ITEM(registration_timeout_us, 1000000u);

  // Whether to verify GPU memory alignment
  CONFIG_ITEM(verify_alignment, true);

  // Required alignment for GPU memory (bytes, typically 256 for GDR)
  CONFIG_ITEM(required_alignment, 256u);
};

/**
 * GPU device information
 */
struct GpuDeviceInfo {
  int deviceId = -1;                  // CUDA device ID
  int pciBusId = -1;                  // PCI bus ID
  int pciDeviceId = -1;               // PCI device ID
  int pciDomainId = 0;                // PCI domain ID
  std::string uuid;                   // Device UUID
  size_t totalMemory = 0;             // Total device memory
  int computeCapabilityMajor = 0;     // Compute capability major version
  int computeCapabilityMinor = 0;     // Compute capability minor version
  bool gdrSupported = false;          // Whether GDR is supported

  bool isValid() const { return deviceId >= 0 && gdrSupported; }
};

/**
 * GPU memory descriptor
 *
 * Contains information needed to identify and access GPU memory,
 * including support for cross-process memory sharing via IPC handles.
 */
struct GpuMemoryDescriptor {
  void* devicePtr = nullptr;          // GPU device pointer
  size_t size = 0;                    // Size in bytes
  int deviceId = -1;                  // CUDA device ID

  // For IPC memory sharing (cross-process GPU memory access)
  struct IpcHandle {
    uint8_t data[64];                 // CUDA IPC memory handle
    bool valid = false;
  };
  IpcHandle ipcHandle;

  // For dmabuf-based registration
  int dmabufFd = -1;                  // dmabuf file descriptor

  // Memory attributes
  bool isManaged = false;             // CUDA managed memory
  bool isPinned = false;              // CUDA pinned (page-locked) memory
  size_t alignment = 0;               // Memory alignment

  bool isValid() const {
    return devicePtr != nullptr && size > 0 && deviceId >= 0;
  }
};

/**
 * GPU memory region registered with IB devices
 */
class GpuMemoryRegion {
 public:
  GpuMemoryRegion() = default;
  ~GpuMemoryRegion();

  // Non-copyable, movable
  GpuMemoryRegion(const GpuMemoryRegion&) = delete;
  GpuMemoryRegion& operator=(const GpuMemoryRegion&) = delete;
  GpuMemoryRegion(GpuMemoryRegion&& other) noexcept;
  GpuMemoryRegion& operator=(GpuMemoryRegion&& other) noexcept;

  /**
   * Register GPU memory with all available IB devices
   *
   * @param desc GPU memory descriptor
   * @param config GDR configuration
   * @return Result containing the registered region or an error
   */
  static Result<std::unique_ptr<GpuMemoryRegion>> create(
      const GpuMemoryDescriptor& desc,
      const GDRConfig& config = GDRConfig());

  /**
   * Register GPU memory using dmabuf mechanism (preferred for cross-process)
   *
   * @param dmabufFd File descriptor for dmabuf
   * @param size Memory size
   * @param deviceId GPU device ID
   * @param config GDR configuration
   * @return Result containing the registered region or an error
   */
  static Result<std::unique_ptr<GpuMemoryRegion>> createFromDmabuf(
      int dmabufFd,
      size_t size,
      int deviceId,
      const GDRConfig& config = GDRConfig());

  // Accessors
  void* devicePtr() const { return desc_.devicePtr; }
  size_t size() const { return desc_.size; }
  int deviceId() const { return desc_.deviceId; }
  const GpuMemoryDescriptor& descriptor() const { return desc_; }

  /**
   * Get the memory region for a specific IB device
   *
   * @param devId IB device ID
   * @return Memory region pointer or nullptr if not registered
   */
  ibv_mr* getMR(int devId) const;

  /**
   * Get the rkey for a specific IB device
   *
   * @param devId IB device ID
   * @return Optional rkey value
   */
  std::optional<uint32_t> getRkey(int devId) const;

  /**
   * Get rkeys for all registered devices
   *
   * @param rkeys Output array to fill with rkeys
   * @return true if all rkeys were obtained successfully
   */
  bool getAllRkeys(std::array<uint32_t, IBDevice::kMaxDeviceCnt>& rkeys) const;

 private:
  GpuMemoryDescriptor desc_;
  std::array<ibv_mr*, IBDevice::kMaxDeviceCnt> mrs_{};
  bool registered_ = false;

  Result<Void> registerWithDevices(const GDRConfig& config);
  Result<Void> registerWithDmabuf(int dmabufFd, const GDRConfig& config);
  void deregister();
};

/**
 * Cache for GPU memory regions
 *
 * Provides efficient caching of registered GPU memory regions to avoid
 * repeated registration overhead.
 */
class GpuMemoryRegionCache {
 public:
  explicit GpuMemoryRegionCache(const GDRConfig& config = GDRConfig());
  ~GpuMemoryRegionCache();

  /**
   * Get or create a memory region for the given descriptor
   *
   * @param desc GPU memory descriptor
   * @return Shared pointer to the memory region
   */
  Result<std::shared_ptr<GpuMemoryRegion>> getOrCreate(
      const GpuMemoryDescriptor& desc);

  /**
   * Invalidate a cached region by device pointer
   *
   * @param devicePtr GPU device pointer
   */
  void invalidate(void* devicePtr);

  /**
   * Clear all cached regions
   */
  void clear();

  /**
   * Get the number of cached regions
   */
  size_t size() const;

 private:
  GDRConfig config_;
  mutable std::mutex mutex_;
  std::unordered_map<void*, std::shared_ptr<GpuMemoryRegion>> cache_;

  void evictIfNeeded();
};

/**
 * GDR Manager - Singleton for global GDR state management
 */
class GDRManager {
 public:
  static GDRManager& instance();

  /**
   * Initialize GDR support
   *
   * @param config GDR configuration
   * @return Result indicating success or failure
   */
  Result<Void> init(const GDRConfig& config);

  /**
   * Shutdown GDR support
   */
  void shutdown();

  /**
   * Check if GDR is initialized and available
   */
  bool isAvailable() const { return initialized_.load(); }

  /**
   * Get information about available GPU devices
   *
   * @return Vector of GPU device information
   */
  const std::vector<GpuDeviceInfo>& getGpuDevices() const { return gpuDevices_; }

  /**
   * Get the memory region cache
   *
   * @return Pointer to cache, or nullptr if GDR is not available
   */
  GpuMemoryRegionCache* getRegionCache() {
    return regionCache_.get();
  }

  /**
   * Check if a specific GPU device supports GDR
   *
   * @param deviceId CUDA device ID
   * @return true if GDR is supported
   */
  bool isGdrSupported(int deviceId) const;

  /**
   * Get the best IB device for a given GPU device (based on PCIe locality)
   *
   * @param gpuDeviceId CUDA device ID
   * @return Optional IB device ID
   */
  std::optional<uint8_t> getBestIBDevice(int gpuDeviceId) const;

  const GDRConfig& config() const { return config_; }

 private:
  GDRManager() = default;
  ~GDRManager();

  GDRManager(const GDRManager&) = delete;
  GDRManager& operator=(const GDRManager&) = delete;

  Result<Void> detectGpuDevices();
  Result<Void> setupGpuIBMapping();

  GDRConfig config_;
  std::atomic<bool> initialized_{false};
  std::vector<GpuDeviceInfo> gpuDevices_;
  std::unique_ptr<GpuMemoryRegionCache> regionCache_;

  // Mapping from GPU device ID to preferred IB device ID
  std::unordered_map<int, uint8_t> gpuToIBMapping_;
};

}  // namespace hf3fs::net
