#pragma once

/**
 * Mock CUDA Runtime for GDR unit tests
 *
 * Provides link-time substitution for CUDA runtime functions.
 * Tests configure behavior via the global singleton before each test
 * and reset after. This allows testing GDR code paths on machines
 * without real GPUs.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

// Minimal CUDA types needed by mock
#ifndef __CUDA_RUNTIME_H__

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInvalidDevice = 10;
constexpr cudaError_t cudaErrorNoDevice = 100;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMapBufferObjectFailed = 14;

enum cudaMemoryType {
  cudaMemoryTypeUnregistered = 0,
  cudaMemoryTypeHost = 1,
  cudaMemoryTypeDevice = 2,
  cudaMemoryTypeManaged = 3,
};

struct cudaPointerAttributes {
  cudaMemoryType type;
  int device;
  void* devicePointer;
  void* hostPointer;
};

struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  int major;
  int minor;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  char uuid[16];
};

struct cudaIpcMemHandle_t {
  uint8_t reserved[64];
};

constexpr unsigned int cudaIpcMemLazyEnablePeerAccess = 0x01;

#endif  // __CUDA_RUNTIME_H__

namespace hf3fs::test {

class MockCudaRuntime {
 public:
  static MockCudaRuntime& instance() {
    static MockCudaRuntime inst;
    return inst;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mu_);
    deviceCount_ = 0;
    deviceProps_.clear();
    mallocBehavior_ = nullptr;
    ipcGetHandleBehavior_ = nullptr;
    ipcOpenHandleBehavior_ = nullptr;
    pointerAttrsBehavior_ = nullptr;
    mallocCalled_ = false;
    mallocCallCount_ = 0;
    freeCalled_.clear();
    setDeviceCalls_.clear();
    syncCalled_ = false;
    syncCallCount_ = 0;
    ipcCloseCalled_ = false;
    ipcCloseCallCount_ = 0;
    fakePointerCounter_ = 0x1000000;  // Start at a non-null address
    nvidiaPeermemLoaded_ = true;
  }

  // Configuration
  void setDeviceCount(int count) {
    std::lock_guard<std::mutex> lock(mu_);
    deviceCount_ = count;
    // Auto-generate default properties
    deviceProps_.clear();
    for (int i = 0; i < count; i++) {
      cudaDeviceProp prop{};
      snprintf(prop.name, sizeof(prop.name), "Mock GPU %d", i);
      prop.totalGlobalMem = 8ULL * 1024 * 1024 * 1024;
      prop.major = 8;
      prop.minor = 0;
      prop.pciBusID = 0x3b + i;
      prop.pciDeviceID = 0;
      prop.pciDomainID = 0;
      deviceProps_[i] = prop;
    }
  }

  void setDeviceProperties(int deviceId, cudaDeviceProp props) {
    std::lock_guard<std::mutex> lock(mu_);
    deviceProps_[deviceId] = props;
  }

  void setMallocBehavior(std::function<cudaError_t(void**, size_t)> fn) {
    std::lock_guard<std::mutex> lock(mu_);
    mallocBehavior_ = std::move(fn);
  }

  void setIpcGetHandleBehavior(std::function<cudaError_t(cudaIpcMemHandle_t*, void*)> fn) {
    std::lock_guard<std::mutex> lock(mu_);
    ipcGetHandleBehavior_ = std::move(fn);
  }

  void setIpcOpenHandleBehavior(
      std::function<cudaError_t(void**, cudaIpcMemHandle_t, unsigned int)> fn) {
    std::lock_guard<std::mutex> lock(mu_);
    ipcOpenHandleBehavior_ = std::move(fn);
  }

  void setPointerAttrsBehavior(
      std::function<cudaError_t(cudaPointerAttributes*, const void*)> fn) {
    std::lock_guard<std::mutex> lock(mu_);
    pointerAttrsBehavior_ = std::move(fn);
  }

  void setNvidiaPeermemLoaded(bool loaded) {
    std::lock_guard<std::mutex> lock(mu_);
    nvidiaPeermemLoaded_ = loaded;
  }

  // Mock implementations
  cudaError_t getDeviceCount(int* count) {
    std::lock_guard<std::mutex> lock(mu_);
    if (deviceCount_ == 0) {
      *count = 0;
      return cudaErrorNoDevice;
    }
    *count = deviceCount_;
    return cudaSuccess;
  }

  cudaError_t getDeviceProperties(cudaDeviceProp* prop, int device) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = deviceProps_.find(device);
    if (it == deviceProps_.end()) return cudaErrorInvalidDevice;
    *prop = it->second;
    return cudaSuccess;
  }

  cudaError_t malloc(void** devPtr, size_t size) {
    std::lock_guard<std::mutex> lock(mu_);
    mallocCalled_ = true;
    mallocCallCount_++;
    if (mallocBehavior_) return mallocBehavior_(devPtr, size);
    // Default: return fake pointer
    fakePointerCounter_ += 0x10000;
    *devPtr = reinterpret_cast<void*>(fakePointerCounter_);
    return cudaSuccess;
  }

  cudaError_t free(void* devPtr) {
    std::lock_guard<std::mutex> lock(mu_);
    freeCalled_.insert(devPtr);
    return cudaSuccess;
  }

  cudaError_t setDevice(int device) {
    std::lock_guard<std::mutex> lock(mu_);
    setDeviceCalls_.push_back(device);
    if (device < 0 || device >= deviceCount_) return cudaErrorInvalidDevice;
    return cudaSuccess;
  }

  cudaError_t deviceSynchronize() {
    std::lock_guard<std::mutex> lock(mu_);
    syncCalled_ = true;
    syncCallCount_++;
    return cudaSuccess;
  }

  cudaError_t ipcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (ipcGetHandleBehavior_) return ipcGetHandleBehavior_(handle, devPtr);
    // Default: fill with deterministic bytes
    for (int i = 0; i < 64; i++) {
      handle->reserved[i] = static_cast<uint8_t>((reinterpret_cast<uintptr_t>(devPtr) + i) & 0xFF);
    }
    return cudaSuccess;
  }

  cudaError_t ipcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    std::lock_guard<std::mutex> lock(mu_);
    if (ipcOpenHandleBehavior_) return ipcOpenHandleBehavior_(devPtr, handle, flags);
    // Default: return new fake pointer
    fakePointerCounter_ += 0x10000;
    *devPtr = reinterpret_cast<void*>(fakePointerCounter_);
    return cudaSuccess;
  }

  cudaError_t ipcCloseMemHandle(void* /*devPtr*/) {
    std::lock_guard<std::mutex> lock(mu_);
    ipcCloseCalled_ = true;
    ipcCloseCallCount_++;
    return cudaSuccess;
  }

  cudaError_t pointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pointerAttrsBehavior_) return pointerAttrsBehavior_(attributes, ptr);
    // Default: report as device memory
    attributes->type = cudaMemoryTypeDevice;
    attributes->device = 0;
    attributes->devicePointer = const_cast<void*>(ptr);
    attributes->hostPointer = nullptr;
    return cudaSuccess;
  }

  // Verification
  bool wasMallocCalled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return mallocCalled_;
  }

  int mallocCallCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return mallocCallCount_;
  }

  bool wasFreeCalled(void* ptr) const {
    std::lock_guard<std::mutex> lock(mu_);
    return freeCalled_.count(ptr) > 0;
  }

  bool wasSetDeviceCalled(int deviceId) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto d : setDeviceCalls_) {
      if (d == deviceId) return true;
    }
    return false;
  }

  int setDeviceCallCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(setDeviceCalls_.size());
  }

  bool wasSyncCalled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return syncCalled_;
  }

  int syncCallCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return syncCallCount_;
  }

  bool wasIpcCloseCalled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return ipcCloseCalled_;
  }

  int ipcCloseCallCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return ipcCloseCallCount_;
  }

  bool isNvidiaPeermemLoaded() const {
    std::lock_guard<std::mutex> lock(mu_);
    return nvidiaPeermemLoaded_;
  }

  int deviceCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return deviceCount_;
  }

 private:
  MockCudaRuntime() { reset(); }
  ~MockCudaRuntime() = default;

  mutable std::mutex mu_;
  int deviceCount_ = 0;
  std::unordered_map<int, cudaDeviceProp> deviceProps_;
  std::function<cudaError_t(void**, size_t)> mallocBehavior_;
  std::function<cudaError_t(cudaIpcMemHandle_t*, void*)> ipcGetHandleBehavior_;
  std::function<cudaError_t(void**, cudaIpcMemHandle_t, unsigned int)> ipcOpenHandleBehavior_;
  std::function<cudaError_t(cudaPointerAttributes*, const void*)> pointerAttrsBehavior_;

  bool mallocCalled_ = false;
  int mallocCallCount_ = 0;
  std::set<void*> freeCalled_;
  std::vector<int> setDeviceCalls_;
  bool syncCalled_ = false;
  int syncCallCount_ = 0;
  bool ipcCloseCalled_ = false;
  int ipcCloseCallCount_ = 0;
  uintptr_t fakePointerCounter_ = 0x1000000;
  bool nvidiaPeermemLoaded_ = true;
};

inline MockCudaRuntime& mockCuda() {
  return MockCudaRuntime::instance();
}

}  // namespace hf3fs::test
