#include "RDMABufAccelerator.h"

#include <cstring>
#include <deque>
#include <mutex>

#include <folly/Likely.h>
#include <folly/experimental/coro/Timeout.h>
#include <folly/fibers/Semaphore.h>
#include <folly/logging/xlog.h>

#ifdef HF3FS_GDR_ENABLED
#include <cuda_runtime.h>
#endif

#include "common/monitor/Recorder.h"

namespace hf3fs::net {

namespace {
monitor::CountRecorder gpuRdmaBufMem("common.ib.gpu_rdma_buf_mem", {}, false);
}  // namespace

// RDMABufAccelerator implementation

RDMABufAccelerator RDMABufAccelerator::createFromGpuPointer(void* devicePtr, size_t size, int deviceId) {
  if (!devicePtr || size == 0 || deviceId < 0) {
    XLOGF(ERR, "Invalid GPU pointer parameters: ptr={}, size={}, device={}",
          devicePtr, size, deviceId);
    return RDMABufAccelerator();
  }

  AcceleratorMemoryDescriptor desc;
  desc.devicePtr = devicePtr;
  desc.size = size;
  desc.deviceId = deviceId;
  desc.dmabufFd = -1;

  return createFromDescriptor(desc);
}

RDMABufAccelerator RDMABufAccelerator::createFromDescriptor(const AcceleratorMemoryDescriptor& desc) {
  if (!desc.isValid()) {
    XLOGF(ERR, "Invalid GPU memory descriptor");
    return RDMABufAccelerator();
  }

  if (!GDRManager::instance().isAvailable()) {
    XLOGF(ERR, "GDR not available");
    return RDMABufAccelerator();
  }

  // Try to get from cache or create new region
  auto* cache = GDRManager::instance().getRegionCache();
  if (!cache) {
    XLOGF(ERR, "GDR region cache not available");
    return RDMABufAccelerator();
  }
  auto result = cache->getOrCreate(desc);
  if (!result) {
    XLOGF(ERR, "Failed to create GPU memory region: {}", result.error().message());
    return RDMABufAccelerator();
  }

  auto region = *result;
  gpuRdmaBufMem.addSample(desc.size);

  return RDMABufAccelerator(region,
                    static_cast<uint8_t*>(desc.devicePtr),
                    desc.size);
}

RDMABufAccelerator RDMABufAccelerator::createFromIpcHandle(const void* ipcHandle, size_t size, int deviceId) {
  if (!ipcHandle || size == 0 || deviceId < 0) {
    XLOGF(ERR, "Invalid IPC handle parameters");
    return RDMABufAccelerator();
  }

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(deviceId);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaSetDevice({}) failed: {}", deviceId, cudaGetErrorString(err));
    return RDMABufAccelerator();
  }

  cudaIpcMemHandle_t cudaHandle;
  std::memcpy(&cudaHandle, ipcHandle, sizeof(cudaHandle));

  void* importedPtr = nullptr;
  err = cudaIpcOpenMemHandle(&importedPtr, cudaHandle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    XLOGF(ERR, "cudaIpcOpenMemHandle failed: {}", cudaGetErrorString(err));
    return RDMABufAccelerator();
  }

  AcceleratorMemoryDescriptor desc;
  desc.devicePtr = importedPtr;
  desc.size = size;
  desc.deviceId = deviceId;
  std::memcpy(desc.ipcHandle.data, &cudaHandle, sizeof(desc.ipcHandle.data));
  desc.ipcHandle.valid = true;

  auto result = createFromDescriptor(desc);
  if (!result.valid()) {
    cudaIpcCloseMemHandle(importedPtr);
    return RDMABufAccelerator();
  }

  auto owner = std::shared_ptr<void>(
      importedPtr,
      [deviceId](void* ptr) {
        if (!ptr) return;
        cudaError_t closeErr = cudaSetDevice(deviceId);
        if (closeErr != cudaSuccess) {
          XLOGF(WARN, "cudaSetDevice({}) failed: {}", deviceId, cudaGetErrorString(closeErr));
        }
        closeErr = cudaIpcCloseMemHandle(ptr);
        if (closeErr != cudaSuccess) {
          XLOGF(WARN, "cudaIpcCloseMemHandle failed: {}", cudaGetErrorString(closeErr));
        }
      });

  result.ipcHandleOwner_ = std::move(owner);
  return result;
#else
  XLOGF(WARN, "IPC handle import requires CUDA runtime - not implemented");
  return RDMABufAccelerator();
#endif
}

RDMARemoteBuf RDMABufAccelerator::toRemoteBuf() const {
  if (!valid()) {
    return RDMARemoteBuf();
  }

  std::array<RDMARemoteBuf::Rkey, IBDevice::kMaxDeviceCnt> rkeys;
  for (auto& rkey : rkeys) {
    rkey.devId = -1;
    rkey.rkey = 0;
  }

  size_t devs = 0;
  for (const auto& dev : IBDevice::all()) {
    if (dev->id() >= IBDevice::kMaxDeviceCnt) continue;

    auto mr = region_->getMR(dev->id());
    if (mr) {
      rkeys[devs].rkey = mr->rkey;
      rkeys[devs].devId = dev->id();
      ++devs;
    }
  }

  if (devs == 0) {
    XLOGF(WARN, "No rkeys available for GPU buffer");
    return RDMARemoteBuf();
  }

  return RDMARemoteBuf(reinterpret_cast<uint64_t>(begin_), length_, rkeys);
}

RDMABufAccelerator RDMABufAccelerator::subrange(size_t offset, size_t length) const {
  if (!valid()) {
    return RDMABufAccelerator();
  }

  if (offset + length > length_) {
    XLOGF(WARN, "Subrange exceeds buffer bounds: offset={}, length={}, size={}",
          offset, length, length_);
    return RDMABufAccelerator();
  }

  return RDMABufAccelerator(region_, begin_ + offset, length);
}

void RDMABufAccelerator::sync(int direction) const {
  if (!valid()) {
    return;
  }

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(region_->deviceId());
  if (err != cudaSuccess) {
    XLOGF(WARN, "cudaSetDevice({}) failed: {}", region_->deviceId(), cudaGetErrorString(err));
    return;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    XLOGF(WARN, "cudaDeviceSynchronize failed: {}", cudaGetErrorString(err));
  }
#else
  (void)direction;
#endif

  XLOGF(DBG, "GPU buffer sync: direction={}, ptr={}, size={}",
        direction, static_cast<void*>(begin_), length_);
}

bool RDMABufAccelerator::getIpcHandle(void* handle) const {
  if (!valid() || !handle) {
    return false;
  }

#ifdef HF3FS_GDR_ENABLED
  cudaError_t err = cudaSetDevice(region_->deviceId());
  if (err != cudaSuccess) {
    XLOGF(WARN, "cudaSetDevice({}) failed: {}", region_->deviceId(), cudaGetErrorString(err));
    return false;
  }
  cudaIpcMemHandle_t* h = static_cast<cudaIpcMemHandle_t*>(handle);
  err = cudaIpcGetMemHandle(h, region_->devicePtr());
  if (err != cudaSuccess) {
    XLOGF(WARN, "cudaIpcGetMemHandle failed: {}", cudaGetErrorString(err));
    return false;
  }
  return true;
#else
  XLOGF(WARN, "IPC handle export requires CUDA runtime - not implemented");
  return false;
#endif
}

// RDMABufAcceleratorPool implementation

class RDMABufAcceleratorPool::Impl {
 public:
  Impl(int deviceId, size_t bufSize, size_t bufCnt)
      : deviceId_(deviceId),
        bufSize_(bufSize),
        sem_(bufCnt) {}

  ~Impl() {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef HF3FS_GDR_ENABLED
    cudaError_t setErr = cudaSetDevice(deviceId_);
    if (setErr != cudaSuccess) {
      XLOGF(WARN, "cudaSetDevice({}) failed: {}", deviceId_, cudaGetErrorString(setErr));
    }
#endif
    for (auto& buf : freeList_) {
      (void)buf;
#ifdef HF3FS_GDR_ENABLED
      if (setErr == cudaSuccess) {
        cudaError_t err = cudaFree(buf);
        if (err != cudaSuccess) {
          XLOGF(WARN, "cudaFree failed: {}", cudaGetErrorString(err));
        }
      }
#endif
      gpuRdmaBufMem.addSample(-static_cast<int64_t>(bufSize_));
    }
    freeList_.clear();
  }

   CoTask<RDMABufAccelerator> allocate(std::optional<folly::Duration> timeout) {
     // Wait for available buffer
     if (UNLIKELY(!sem_.try_wait())) {
       if (timeout.has_value()) {
         auto result = co_await folly::coro::co_awaitTry(
             folly::coro::timeout(sem_.co_wait(), timeout.value()));
         if (result.hasException()) {
           co_return RDMABufAccelerator();
         }
       } else {
         co_await sem_.co_wait();
       }
     }

     // Try to get from free list
     {
       std::lock_guard<std::mutex> lock(mutex_);
       if (!freeList_.empty()) {
         void* ptr = freeList_.back();
         freeList_.pop_back();
         co_return RDMABufAccelerator::createFromGpuPointer(ptr, bufSize_, deviceId_);
       }
     }

     // Allocate new GPU memory
 #ifdef HF3FS_GDR_ENABLED
     void* ptr = nullptr;
     cudaError_t err = cudaSetDevice(deviceId_);
     if (err != cudaSuccess) {
       XLOGF(WARN, "cudaSetDevice({}) failed: {}", deviceId_, cudaGetErrorString(err));
       sem_.signal();
       co_return RDMABufAccelerator();
     }
     err = cudaMalloc(&ptr, bufSize_);
     if (err != cudaSuccess) {
       XLOGF(WARN, "cudaMalloc failed: {}", cudaGetErrorString(err));
       sem_.signal();
       co_return RDMABufAccelerator();
     }

     gpuRdmaBufMem.addSample(bufSize_);
     co_return RDMABufAccelerator::createFromGpuPointer(ptr, bufSize_, deviceId_);
 #else
     XLOGF(WARN, "GPU memory allocation requires CUDA runtime");
     sem_.signal();  // Return the semaphore token since we failed
     co_return RDMABufAccelerator();
 #endif
   }

  void deallocate(void* ptr) {
    if (!ptr) return;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      freeList_.push_back(ptr);
    }
    sem_.signal();
  }

  size_t freeCnt() const { return sem_.getAvailableTokens(); }

 private:
  int deviceId_;
  size_t bufSize_;
  folly::fibers::Semaphore sem_;
  std::mutex mutex_;
  std::deque<void*> freeList_;
};

std::shared_ptr<RDMABufAcceleratorPool> RDMABufAcceleratorPool::create(
     int deviceId, size_t bufSize, size_t bufCnt) {
   return std::shared_ptr<RDMABufAcceleratorPool>(
       new RDMABufAcceleratorPool(deviceId, bufSize, bufCnt));
 }

 RDMABufAcceleratorPool::RDMABufAcceleratorPool(int deviceId, size_t bufSize, size_t bufCnt)
     : deviceId_(deviceId),
       bufSize_(bufSize),
       bufCnt_(bufCnt),
       impl_(std::make_unique<Impl>(deviceId, bufSize, bufCnt)) {}

 RDMABufAcceleratorPool::~RDMABufAcceleratorPool() = default;

 CoTask<RDMABufAccelerator> RDMABufAcceleratorPool::allocate(std::optional<folly::Duration> timeout) {
   co_return co_await impl_->allocate(timeout);
 }

 size_t RDMABufAcceleratorPool::freeCnt() const {
   return impl_->freeCnt();
 }

 void RDMABufAcceleratorPool::deallocate(void* ptr) {
   impl_->deallocate(ptr);
 }

}  // namespace hf3fs::net
