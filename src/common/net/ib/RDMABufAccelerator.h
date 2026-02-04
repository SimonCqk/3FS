#pragma once

#include <memory>
#include <optional>

#include "common/net/ib/AcceleratorMemory.h"
#include "common/net/ib/RDMABuf.h"

namespace hf3fs::net {

/**
 * RDMA buffer wrapper for GPU memory
 *
 * RDMABufAccelerator extends the RDMABuf concept to support GPU device memory.
 * It handles GPU memory registration with IB devices for direct RDMA
 * transfers (GPU Direct RDMA).
 *
 * Key differences from RDMABuf:
 * - Memory is allocated on GPU device (not host)
 * - Uses nvidia_peermem for memory registration
 * - May require synchronization between GPU and RDMA operations
 * - Supports IPC memory sharing for cross-process scenarios
 */
class RDMABufAccelerator {
 public:
   RDMABufAccelerator() = default;

   // Non-copyable but movable
   RDMABufAccelerator(const RDMABufAccelerator&) = delete;
   RDMABufAccelerator& operator=(const RDMABufAccelerator&) = delete;
   RDMABufAccelerator(RDMABufAccelerator&&) noexcept = default;
   RDMABufAccelerator& operator=(RDMABufAccelerator&&) noexcept = default;

  /**
   * Create from existing GPU device pointer
   *
   * The caller retains ownership of the GPU memory.
   * The GPU memory must remain valid for the lifetime of this object.
   *
   * @param devicePtr GPU device pointer
   * @param size Size of the memory region
   * @param deviceId CUDA device ID
   * @return The created buffer, or invalid buffer on failure
   */
   static RDMABufAccelerator createFromGpuPointer(void* devicePtr, size_t size, int deviceId);

  /**
   * Create from GPU memory descriptor
   *
   * @param desc GPU memory descriptor with all necessary information
   * @return The created buffer, or invalid buffer on failure
   */
   static RDMABufAccelerator createFromDescriptor(const AcceleratorMemoryDescriptor& desc);

  /**
   * Create from IPC handle (cross-process GPU memory sharing)
   *
   * @param ipcHandle CUDA IPC memory handle
   * @param size Expected size of the memory
   * @param deviceId CUDA device ID to use for import
   * @return The created buffer, or invalid buffer on failure
   */
   static RDMABufAccelerator createFromIpcHandle(const void* ipcHandle, size_t size, int deviceId);

  /**
   * Check if the buffer is valid and usable
   */
  bool valid() const { return region_ != nullptr; }
  explicit operator bool() const { return valid(); }

  /**
   * Get the GPU device pointer
   */
  void* devicePtr() const {
    return region_ ? const_cast<void*>(region_->devicePtr()) : nullptr;
  }

  /**
   * Get a pointer to the data (same as devicePtr for GPU buffers)
   */
  uint8_t* ptr() { return static_cast<uint8_t*>(devicePtr()); }
  const uint8_t* ptr() const { return static_cast<const uint8_t*>(devicePtr()); }

  /**
   * Get the total capacity of the buffer
   */
  size_t capacity() const { return region_ ? region_->size() : 0; }

  /**
   * Get the current size of the buffer (may be less than capacity)
   */
  size_t size() const { return length_; }

  /**
   * Check if the buffer is empty
   */
  bool empty() const { return size() == 0; }

  /**
   * Get the GPU device ID
   */
  int deviceId() const { return region_ ? region_->deviceId() : -1; }

  /**
   * Get the memory region for a specific IB device
   *
   * @param devId IB device ID
   * @return Memory region pointer or nullptr
   */
  ibv_mr* getMR(int devId) const {
    return region_ ? region_->getMR(devId) : nullptr;
  }

  /**
   * Get the rkey for a specific IB device
   */
  std::optional<uint32_t> getRkey(int devId) const {
    return region_ ? region_->getRkey(devId) : std::nullopt;
  }

  /**
   * Convert to RDMARemoteBuf for remote RDMA operations
   *
   * The returned RDMARemoteBuf contains the device address and rkeys
   * needed for remote RDMA read/write operations.
   */
  RDMARemoteBuf toRemoteBuf() const;

  /**
   * Reset the buffer range to full capacity
   */
  void resetRange() {
    if (region_) {
      begin_ = static_cast<uint8_t*>(const_cast<void*>(region_->devicePtr()));
      length_ = region_->size();
    }
  }

  /**
   * Advance the start pointer by n bytes
   * @return false if n > size()
   */
  bool advance(size_t n) {
    if (n > length_) return false;
    begin_ += n;
    length_ -= n;
    return true;
  }

  /**
   * Reduce the size by n bytes from the end
   * @return false if n > size()
   */
  bool subtract(size_t n) {
    if (n > length_) return false;
    length_ -= n;
    return true;
  }

   /**
    * Create a subrange view of this buffer
    */
   RDMABufAccelerator subrange(size_t offset, size_t length) const;

   /**
    * Get the first `length` bytes
    */
   RDMABufAccelerator first(size_t length) const { return subrange(0, length); }

   /**
    * Take the first `length` bytes (modifies this buffer)
    */
   RDMABufAccelerator takeFirst(size_t length) {
     auto buf = first(length);
     advance(length);
     return buf;
   }

   /**
    * Get the last `length` bytes
    */
   RDMABufAccelerator last(size_t length) const {
     if (length > length_) return RDMABufAccelerator();
     return subrange(length_ - length, length);
   }

   /**
    * Take the last `length` bytes (modifies this buffer)
    */
   RDMABufAccelerator takeLast(size_t length) {
     auto buf = last(length);
     subtract(length);
     return buf;
   }

  /**
   * Check if a pointer range is contained within this buffer
   */
  bool contains(const uint8_t* data, uint32_t len) const {
    return ptr() <= data && data + len <= ptr() + capacity();
  }

  /**
   * Synchronize GPU memory for RDMA operations
   *
   * @param direction 0 = before RDMA (ensure GPU writes visible to RDMA)
   *                  1 = after RDMA (ensure RDMA writes visible to GPU)
   */
  void sync(int direction) const;

  /**
   * Get IPC handle for sharing this buffer with other processes
   *
   * @param handle Output buffer for the IPC handle (64 bytes)
   * @return true if successful
   */
  bool getIpcHandle(void* handle) const;

  private:
   RDMABufAccelerator(std::shared_ptr<AcceleratorMemoryRegion> region, uint8_t* begin, size_t length)
       : region_(std::move(region)),
         begin_(begin),
         length_(length) {}

  std::shared_ptr<AcceleratorMemoryRegion> region_;
  uint8_t* begin_ = nullptr;
  size_t length_ = 0;
 std::shared_ptr<void> ipcHandleOwner_;
};

/**
 * Pool for GPU RDMA buffers
 *
 * Similar to RDMABufPool but for GPU memory.
 * Pre-allocates GPU buffers for efficient reuse.
 */
class RDMABufAcceleratorPool : public std::enable_shared_from_this<RDMABufAcceleratorPool> {
 public:
   /**
    * Create a new GPU buffer pool
    *
    * @param deviceId CUDA device ID for buffer allocation
    * @param bufSize Size of each buffer
    * @param bufCnt Number of buffers in the pool
    * @return Shared pointer to the pool
    */
   static std::shared_ptr<RDMABufAcceleratorPool> create(int deviceId, size_t bufSize, size_t bufCnt);

   ~RDMABufAcceleratorPool();

   /**
    * Allocate a buffer from the pool
    *
    * @param timeout Optional timeout for waiting (nullptr = no timeout)
    * @return Allocated buffer, or invalid buffer on timeout/failure
    */
   CoTask<RDMABufAccelerator> allocate(std::optional<folly::Duration> timeout = std::nullopt);

  /**
   * Get buffer size for this pool
   */
  size_t bufSize() const { return bufSize_; }

  /**
   * Get number of free buffers
   */
  size_t freeCnt() const;

  /**
   * Get total number of buffers
   */
  size_t totalCnt() const { return bufCnt_; }

  /**
   * Get the CUDA device ID for this pool
   */
  int deviceId() const { return deviceId_; }

  private:
   RDMABufAcceleratorPool(int deviceId, size_t bufSize, size_t bufCnt);

  void deallocate(void* ptr);

  int deviceId_;
  size_t bufSize_;
  size_t bufCnt_;

  // Internal implementation
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Unified RDMA buffer that can hold either host or GPU memory
 *
 * This is a variant type that can represent either a regular RDMABuf
 * (host memory) or an RDMABufAccelerator (GPU memory), providing a uniform
 * interface for code that needs to handle both.
 */
class RDMABufUnified {
 public:
   enum class Type {
     Empty,
     Host,
     Gpu,
   };

   RDMABufUnified() : type_(Type::Empty) {}

   explicit RDMABufUnified(RDMABuf hostBuf)
       : type_(Type::Host),
         hostBuf_(std::move(hostBuf)) {}

   explicit RDMABufUnified(RDMABufAccelerator gpuBuf)
       : type_(Type::Gpu),
         gpuBuf_(std::move(gpuBuf)) {}

   Type type() const { return type_; }
   bool isHost() const { return type_ == Type::Host; }
   bool isGpu() const { return type_ == Type::Gpu; }
   bool valid() const {
     switch (type_) {
       case Type::Host: return hostBuf_.valid();
       case Type::Gpu: return gpuBuf_.valid();
       default: return false;
     }
   }

   explicit operator bool() const { return valid(); }

   // Access the underlying buffer (caller must check type first)
   RDMABuf& asHost() { return hostBuf_; }
   const RDMABuf& asHost() const { return hostBuf_; }
   RDMABufAccelerator& asGpu() { return gpuBuf_; }
   const RDMABufAccelerator& asGpu() const { return gpuBuf_; }

  // Common operations that work for both types
  uint8_t* ptr() {
    return isHost() ? hostBuf_.ptr() : isGpu() ? gpuBuf_.ptr() : nullptr;
  }
  const uint8_t* ptr() const {
    return isHost() ? hostBuf_.ptr() : isGpu() ? gpuBuf_.ptr() : nullptr;
  }

  size_t size() const {
    return isHost() ? hostBuf_.size() : isGpu() ? gpuBuf_.size() : 0;
  }

  size_t capacity() const {
    return isHost() ? hostBuf_.capacity() : isGpu() ? gpuBuf_.capacity() : 0;
  }

  ibv_mr* getMR(int devId) const {
    return isHost() ? hostBuf_.getMR(devId) : isGpu() ? gpuBuf_.getMR(devId) : nullptr;
  }

  RDMARemoteBuf toRemoteBuf() const {
    return isHost() ? hostBuf_.toRemoteBuf() : isGpu() ? gpuBuf_.toRemoteBuf() : RDMARemoteBuf();
  }

  private:
   Type type_;
   RDMABuf hostBuf_;
   RDMABufAccelerator gpuBuf_;
};

}  // namespace hf3fs::net
