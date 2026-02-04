#include "AcceleratorMemory.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <folly/logging/xlog.h>

#ifdef HF3FS_GDR_ENABLED
#include <cuda_runtime.h>
#endif

#include "common/monitor/Recorder.h"

// Forward declarations for CUDA functions (loaded dynamically to avoid hard dependency)
// In production, these would be loaded via dlopen/dlsym or linked conditionally

namespace hf3fs::net {

namespace {

monitor::CountRecorder gdrMemRegistered("common.ib.gdr_mem_registered", {}, false);
monitor::CountRecorder gdrMemCached("common.ib.gdr_mem_cached", {}, false);

// Access flags for GDR memory registration
constexpr int kGDRAccessFlags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
    IBV_ACCESS_RELAXED_ORDERING;

}  // namespace

// AcceleratorMemoryRegion implementation

AcceleratorMemoryRegion::~AcceleratorMemoryRegion() {
  deregister();
}

AcceleratorMemoryRegion::AcceleratorMemoryRegion(AcceleratorMemoryRegion&& other) noexcept
    : desc_(other.desc_),
      mrs_(other.mrs_),
      registered_(other.registered_) {
  other.mrs_.fill(nullptr);
  other.registered_ = false;
}

AcceleratorMemoryRegion& AcceleratorMemoryRegion::operator=(AcceleratorMemoryRegion&& other) noexcept {
  if (this != &other) {
    deregister();
    desc_ = other.desc_;
    mrs_ = other.mrs_;
    registered_ = other.registered_;
    other.mrs_.fill(nullptr);
    other.registered_ = false;
  }
  return *this;
}

Result<std::unique_ptr<AcceleratorMemoryRegion>> AcceleratorMemoryRegion::create(
    const AcceleratorMemoryDescriptor& desc,
    const GDRConfig& config) {
  if (!desc.isValid()) {
    return makeError(StatusCode::kInvalidArg, "Invalid GPU memory descriptor");
  }

  // Check alignment if configured
  if (config.verify_alignment()) {
    auto alignment = reinterpret_cast<uintptr_t>(desc.devicePtr) % config.required_alignment();
    if (alignment != 0) {
      XLOGF(WARN,
            "GPU memory address {} is not aligned to {} bytes (offset: {})",
            desc.devicePtr,
            config.required_alignment(),
            alignment);
    }
  }

  auto region = std::make_unique<AcceleratorMemoryRegion>();
   region->desc_ = desc;

  // Prefer dmabuf if available and configured
  if (config.use_dmabuf() && desc.dmabufFd >= 0) {
    auto result = region->registerWithDmabuf(desc.dmabufFd, config);
    if (!result) {
      return makeError(result.error());
    }
  } else {
    auto result = region->registerWithDevices(config);
    if (!result) {
      return makeError(result.error());
    }
  }

  gdrMemRegistered.addSample(desc.size);
  return std::move(region);
}

Result<std::unique_ptr<AcceleratorMemoryRegion>> AcceleratorMemoryRegion::createFromDmabuf(
    int dmabufFd,
    size_t size,
    int deviceId,
    const GDRConfig& config) {
   if (dmabufFd < 0 || size == 0) {
     return makeError(StatusCode::kInvalidArg, "Invalid dmabuf parameters");
   }

   auto region = std::make_unique<AcceleratorMemoryRegion>();
  region->desc_.dmabufFd = dmabufFd;
  region->desc_.size = size;
  region->desc_.deviceId = deviceId;

  auto result = region->registerWithDmabuf(dmabufFd, config);
  if (!result) {
    return makeError(result.error());
  }

  gdrMemRegistered.addSample(size);
  return std::move(region);
}

ibv_mr* AcceleratorMemoryRegion::getMR(int devId) const {
   if (devId < 0 || static_cast<size_t>(devId) >= mrs_.size()) {
     return nullptr;
   }
   return mrs_[devId];
 }

 std::optional<uint32_t> AcceleratorMemoryRegion::getRkey(int devId) const {
   auto mr = getMR(devId);
   if (mr) {
     return mr->rkey;
   }
   return std::nullopt;
 }

 bool AcceleratorMemoryRegion::getAllRkeys(
     std::array<uint32_t, IBDevice::kMaxDeviceCnt>& rkeys) const {
  bool hasAny = false;
  for (size_t i = 0; i < mrs_.size(); ++i) {
    if (mrs_[i]) {
      rkeys[i] = mrs_[i]->rkey;
      hasAny = true;
    } else {
      rkeys[i] = 0;
    }
  }
  return hasAny;
}

Result<Void> AcceleratorMemoryRegion::registerWithDevices(const GDRConfig& config) {
  (void)config;
  if (!IBManager::initialized()) {
    return makeError(RPCCode::kIBDeviceNotInitialized, "IB not initialized");
  }

  size_t registeredCount = 0;
  for (const auto& dev : IBDevice::all()) {
    if (dev->id() >= IBDevice::kMaxDeviceCnt) {
      XLOGF(ERR, "Device ID {} exceeds maximum {}", dev->id(), IBDevice::kMaxDeviceCnt);
      continue;
    }

    // For GDR, we use ibv_reg_mr with the GPU pointer directly
    // The underlying driver (nvidia_peermem) handles the GPU memory
    auto mr = dev->regMemory(desc_.devicePtr, desc_.size, kGDRAccessFlags);
    if (!mr) {
      XLOGF(WARN,
            "Failed to register GPU memory {} (size: {}) with IB device {}",
            desc_.devicePtr,
            desc_.size,
            dev->name());
      // Continue trying other devices
      continue;
    }

    mrs_[dev->id()] = mr;
    ++registeredCount;
    XLOGF(DBG,
          "Registered GPU memory {} (size: {}) with IB device {}, lkey: {}, rkey: {}",
          desc_.devicePtr,
          desc_.size,
          dev->name(),
          mr->lkey,
          mr->rkey);
  }

  if (registeredCount == 0) {
    return makeError(StatusCode::kIOError, "Failed to register GPU memory with any IB device");
  }

  registered_ = true;
  return Void{};
}

Result<Void> AcceleratorMemoryRegion::registerWithDmabuf(int dmabufFd, const GDRConfig& config) {
  (void)config;
  if (!IBManager::initialized()) {
    return makeError(RPCCode::kIBDeviceNotInitialized, "IB not initialized");
  }

  if (dmabufFd < 0) {
    return makeError(StatusCode::kInvalidArg, "Invalid dmabuf fd");
  }

#ifdef IBV_ACCESS_OPTIONAL_RANGE
  // Use ibv_reg_dmabuf_mr if available (RDMA-core >= 34)
  // This is the preferred method for GDR as it properly handles
  // GPU memory ownership and lifetime

  size_t registeredCount = 0;
  for (const auto& dev : IBDevice::all()) {
    if (dev->id() >= IBDevice::kMaxDeviceCnt) {
      continue;
    }

    // ibv_reg_dmabuf_mr(pd, offset, length, iova, dmabuf_fd, access)
    // offset: offset into dmabuf
    // length: length to register
    // iova: I/O virtual address (can be 0 for GDR)
    auto mr = ibv_reg_dmabuf_mr(
        dev->pd(),
        0,  // offset
        desc_.size,
        0,  // iova
        dmabufFd,
        kGDRAccessFlags);

    if (!mr) {
      XLOGF(WARN,
            "Failed to register dmabuf (fd: {}, size: {}) with IB device {}: {}",
            dmabufFd,
            desc_.size,
            dev->name(),
            strerror(errno));
      continue;
    }

    mrs_[dev->id()] = mr;
    ++registeredCount;
    XLOGF(DBG,
          "Registered dmabuf (fd: {}, size: {}) with IB device {}, lkey: {}, rkey: {}",
          dmabufFd,
          desc_.size,
          dev->name(),
          mr->lkey,
          mr->rkey);
  }

  if (registeredCount == 0) {
    return makeError(StatusCode::kIOError, "Failed to register dmabuf with any IB device");
  }

  registered_ = true;
  return Void{};
#else
  // Fallback: dmabuf registration not available
  // Only fall back to direct registration if we have a valid GPU pointer
  if (!desc_.devicePtr) {
    XLOGF(ERR, "ibv_reg_dmabuf_mr not available and no valid GPU pointer for fallback");
    return makeError(StatusCode::kNotSupported,
                     "dmabuf registration requires ibv_reg_dmabuf_mr or valid GPU pointer");
  }
  XLOGF(WARN, "ibv_reg_dmabuf_mr not available, falling back to direct registration");
  return registerWithDevices(config);
#endif
}

void AcceleratorMemoryRegion::deregister() {
  if (!registered_) {
    return;
  }

  for (size_t i = 0; i < mrs_.size(); ++i) {
    if (mrs_[i]) {
      auto dev = IBDevice::get(i);
      if (dev) {
        dev->deregMemory(mrs_[i]);
      }
      mrs_[i] = nullptr;
    }
  }

  if (desc_.size > 0) {
    gdrMemRegistered.addSample(-static_cast<int64_t>(desc_.size));
  }

  registered_ = false;
  XLOGF(DBG, "Deregistered GPU memory region {}", desc_.devicePtr);
}

// AcceleratorMemoryRegionCache implementation

AcceleratorMemoryRegionCache::AcceleratorMemoryRegionCache(const GDRConfig& config)
    : config_(config) {}

AcceleratorMemoryRegionCache::~AcceleratorMemoryRegionCache() {
  clear();
}

Result<std::shared_ptr<AcceleratorMemoryRegion>> AcceleratorMemoryRegionCache::getOrCreate(
    const AcceleratorMemoryDescriptor& desc) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check cache first
  auto it = cache_.find(desc.devicePtr);
  if (it != cache_.end()) {
    auto& cached = it->second;
    // Verify the cached region matches
    if (cached->size() >= desc.size && cached->deviceId() == desc.deviceId) {
      return cached;
    }
    // Size mismatch, remove and recreate
    cache_.erase(it);
  }

  // Evict if needed
  evictIfNeeded();

  // Create new region
   auto result = AcceleratorMemoryRegion::create(desc, config_);
   if (!result) {
     return makeError(result.error());
   }

   auto region = std::shared_ptr<AcceleratorMemoryRegion>(std::move(*result));
  cache_[desc.devicePtr] = region;
  gdrMemCached.addSample(1);

  return region;
}

void AcceleratorMemoryRegionCache::invalidate(void* devicePtr) {
   std::lock_guard<std::mutex> lock(mutex_);
   auto it = cache_.find(devicePtr);
   if (it != cache_.end()) {
     cache_.erase(it);
     gdrMemCached.addSample(-1);
   }
 }

 void AcceleratorMemoryRegionCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto count = cache_.size();
  cache_.clear();
  if (count > 0) {
    gdrMemCached.addSample(-static_cast<int64_t>(count));
  }
}

size_t AcceleratorMemoryRegionCache::size() const {
   std::lock_guard<std::mutex> lock(mutex_);
   return cache_.size();
 }

 void AcceleratorMemoryRegionCache::evictIfNeeded() {
  // Simple LRU-like eviction: if cache is full, remove oldest entries
  // In a production implementation, we'd use a proper LRU cache
  while (cache_.size() >= config_.max_cached_regions()) {
    // Remove first entry (arbitrary in unordered_map, but simple)
    auto it = cache_.begin();
    if (it != cache_.end()) {
      cache_.erase(it);
      gdrMemCached.addSample(-1);
    }
  }
}

// GDRManager implementation

GDRManager& GDRManager::instance() {
  static GDRManager instance;
  return instance;
}

GDRManager::~GDRManager() {
  shutdown();
}

Result<Void> GDRManager::init(const GDRConfig& config) {
  if (initialized_.load()) {
    return Void{};
  }

  config_ = config;

  // Parse HF3FS_GDR_ENABLED env var
  const char* gdrEnabledEnv = std::getenv("HF3FS_GDR_ENABLED");
  if (gdrEnabledEnv) {
    if (std::string(gdrEnabledEnv) == "0") {
      XLOGF(INFO, "GDR disabled by HF3FS_GDR_ENABLED=0");
      return Void{};  // Return early, don't initialize
    }
    // "1" means require GDR - will fail later if detection fails
  }

  // Parse HF3FS_GDR_FALLBACK env var
  fallbackMode_ = FallbackMode::Auto;  // default
  const char* fallbackEnv = std::getenv("HF3FS_GDR_FALLBACK");
  if (fallbackEnv) {
    std::string fallback(fallbackEnv);
    if (fallback == "host") {
      fallbackMode_ = FallbackMode::Host;
    } else if (fallback == "fail") {
      fallbackMode_ = FallbackMode::Fail;
    }
    // "auto" or unrecognized → default Auto
  }

  if (!config_.enabled()) {
    XLOGF(INFO, "GDR support is disabled by configuration");
    return Void{};
  }

  // Detect GPU devices
  auto detectResult = detectGpuDevices();
  if (!detectResult) {
    XLOGF(WARN, "Failed to detect GPU devices: {}", detectResult.error().message());
    // Continue without GDR support
    return Void{};
  }

  if (gpuDevices_.empty()) {
    XLOGF(INFO, "No GPU devices found, GDR support unavailable");
    return Void{};
  }

  // Setup GPU to IB device mapping
  auto mappingResult = setupGpuIBMapping();
  if (!mappingResult) {
    XLOGF(WARN, "Failed to setup GPU-IB mapping: {}", mappingResult.error().message());
  }

  // Initialize region cache
  regionCache_ = std::make_unique<AcceleratorMemoryRegionCache>(config_);

  initialized_.store(true);
  XLOGF(INFO, "GDR support initialized with {} GPU device(s)", gpuDevices_.size());

  return Void{};
}

void GDRManager::shutdown() {
  if (!initialized_.load()) {
    return;
  }

  if (regionCache_) {
    regionCache_->clear();
    regionCache_.reset();
  }

  gpuDevices_.clear();
  gpuToIBMapping_.clear();
  initialized_.store(false);

  XLOGF(INFO, "GDR support shut down");
}

bool GDRManager::isGdrSupported(int deviceId) const {
  if (!initialized_.load()) {
    return false;
  }

  for (const auto& dev : gpuDevices_) {
    if (dev.deviceId == deviceId) {
      return dev.gdrSupported;
    }
  }
  return false;
}

std::optional<uint8_t> GDRManager::getBestIBDevice(int gpuDeviceId) const {
  auto it = gpuToIBMapping_.find(gpuDeviceId);
  if (it != gpuToIBMapping_.end()) {
    return it->second;
  }
  // Return first available IB device as fallback
  if (!IBDevice::all().empty()) {
    return IBDevice::all()[0]->id();
  }
  return std::nullopt;
}

Result<Void> GDRManager::detectGpuDevices() {
  // This is a placeholder implementation.
  // In production, this would use CUDA runtime API (loaded dynamically) to:
  // 1. Call cudaGetDeviceCount() to get number of devices
  // 2. For each device, call cudaGetDeviceProperties() to get properties
  // 3. Check if device supports GPU Direct RDMA

  // For now, we check if nvidia_peermem module is loaded (indicates GDR support)
  // and try to detect devices via sysfs

  // Check for nvidia_peermem module
  int fd = open("/sys/module/nvidia_peermem", O_RDONLY | O_DIRECTORY);
  if (fd < 0) {
    XLOGF(INFO, "nvidia_peermem module not loaded, GDR may not be available");
  } else {
    close(fd);
    XLOGF(DBG, "nvidia_peermem module detected");
  }

  // Try to detect NVIDIA GPUs via sysfs
  // /sys/bus/pci/drivers/nvidia/
  const char* nvidiaSysfsPath = "/sys/bus/pci/drivers/nvidia";
  fd = open(nvidiaSysfsPath, O_RDONLY | O_DIRECTORY);
  if (fd >= 0) {
    close(fd);
    // In production, we would enumerate devices here
    // For now, assume CUDA runtime will be used to get accurate device info

    // Create a placeholder device entry
    // The actual device detection would be done via CUDA API
    XLOGF(INFO, "NVIDIA GPU driver detected, GDR support may be available");
  }

  // Note: Actual GPU device detection will be done when CUDA context is created
  // by the user application. The GDR registration will fail gracefully if
  // the device doesn't support GDR.

  return Void{};
}

Result<Void> GDRManager::setupGpuIBMapping() {
   // Setup mapping between GPU devices and IB devices based on PCIe topology
   // The goal is to find the IB device that has the shortest PCIe path to each GPU

   // This is a simplified implementation that assumes GPUs and IB devices
   // might be on the same PCIe switch if they share the same PCIe domain

   // In production, we would parse the PCIe topology from sysfs or use
   // nvidia-smi topology information

   // For now, create a simple round-robin mapping if no topology info available
   const auto& ibDevices = IBDevice::all();
   if (ibDevices.empty()) {
     XLOGF(WARN, "No IB devices available for GPU-IB mapping");
     return Void{};
   }

   for (size_t i = 0; i < gpuDevices_.size(); ++i) {
     // Simple round-robin assignment
     gpuToIBMapping_[gpuDevices_[i].deviceId] = ibDevices[i % ibDevices.size()]->id();
   }

   XLOGF(DBG, "GPU-IB mapping established for {} GPU(s)", gpuToIBMapping_.size());
   return Void{};
}

// detectMemoryType implementation

MemoryType detectMemoryType(const void* ptr) {
  if (!ptr) {
    return MemoryType::Unknown;
  }

#ifdef HF3FS_GDR_ENABLED
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  
  if (err != cudaSuccess) {
    // Not a CUDA-known pointer, treat as host memory
    cudaGetLastError();  // Clear the error
    return MemoryType::Host;
  }

  switch (attrs.type) {
    case cudaMemoryTypeDevice:
      return MemoryType::Device;
    case cudaMemoryTypeManaged:
      return MemoryType::Managed;
    case cudaMemoryTypeHost:
      // Check if it's pinned (page-locked) host memory
      if (attrs.devicePointer != nullptr) {
        return MemoryType::Pinned;
      }
      return MemoryType::Host;
    default:
      return MemoryType::Host;
  }
#else
  // Without GDR support, all pointers are treated as host memory
  return MemoryType::Host;
#endif
}

}  // namespace hf3fs::net
