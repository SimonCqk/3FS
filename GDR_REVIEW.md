GDR Integration Review Notes
============================

Scope
-----
- Branch: `origin/features/gdr-integration`
- Commit: `5d193063aeec4cc2e5291c37c62be5865c0b53ac`
- Topic: usrbio GDR enablement, GPU memory registration, IPC import/export, IO path

Findings and Fixes
------------------

### CRITICAL - FIXED

1. **GDR init can exit early without creating `regionCache_`**
   - Issue: APIs unconditionally accessed `GDRManager::getRegionCache()` which
     could crash or proceed incorrectly when GDR is unavailable.
   - Fix: Changed `getRegionCache()` to return pointer (nullable), added null
     checks before all cache access in UsrbIoGdr.cc, and enhanced
     `hf3fs_gdr_available()` to verify cache exists.
   - Files: `src/common/net/ib/GpuMemory.h`, `src/lib/api/UsrbIoGdr.cc`

2. **IPC export/import returned success with placeholder/nullptr**
   - Issue: Export packed only metadata; import returned success with
     `importedPtr == nullptr`, creating invalid iovs.
   - Fix: Both functions now return `-ENOTSUP` when CUDA runtime is unavailable,
     failing fast instead of creating unusable handles.
   - Files: `src/lib/api/UsrbIoGdr.cc`

3. **StorageClient CPU operations on GPU memory**
   - Issue: Read/write paths used CPU `memcpy`/checksum/inline data directly on
     `IOBuffer` pointers, which would crash on GPU device memory.
   - Fix:
     - Added `isGpuMemory()` flag to `IOBuffer` class
     - Added `registerGpuIOBuffer()` method to StorageClient
     - Modified `buildBatchRequest` for reads to disable inline data for GPU buffers
     - Modified read response handling to skip CPU memcpy/checksum for GPU buffers
     - Modified write path to skip CPU checksum and inline data for GPU buffers
   - Files: `src/client/storage/StorageClient.h`, `src/client/storage/StorageClient.cc`,
     `src/client/storage/StorageClientImpl.cc`

### HIGH - FIXED

4. **dmabuf registration fallback with no valid GPU pointer**
   - Issue: When `ibv_reg_dmabuf_mr` is unavailable, fallback to
     `registerWithDevices()` would fail silently or crash if `devicePtr` is null.
   - Fix: Explicitly check for valid `devicePtr` before fallback; return error
     if neither dmabuf nor direct registration is possible.
   - Files: `src/common/net/ib/GpuMemory.cc`

### MEDIUM - FIXED

5. **GPU MR cache not invalidated on memory free**
   - Issue: Pointer reuse after GPU memory free could leave stale MRs/rkeys.
   - Fix: Added `cache->invalidate(devicePtr)` call in `hf3fs_iovdestroy_gpu()`
     before releasing the handle.
   - Files: `src/lib/api/UsrbIoGdr.cc`

6. **GpuShmBuf wrapped GPU pointers into host RDMABuf**
   - Issue: `memh()` used `RDMABuf::createFromUserBuffer()` without GPU flag,
     re-entering the CPU-only StorageClient path.
   - Fix: Create `IOBuffer` with `isGpuMemory=true` to ensure GPU-aware handling.
   - Files: `src/lib/common/GpuShm.h`, `src/lib/common/GpuShm.cc`

Summary of Changes
------------------

### src/common/net/ib/GpuMemory.h
- Changed `getRegionCache()` to return `GpuMemoryRegionCache*` (nullable)

### src/common/net/ib/GpuMemory.cc
- Added null check for `devicePtr` in dmabuf fallback path

### src/lib/api/UsrbIoGdr.cc
- Enhanced `hf3fs_gdr_available()` to check cache existence
- Added null checks for `getRegionCache()` in all iov create/open/wrap functions
- Changed `hf3fs_iov_export_gpu()` to return `-ENOTSUP` when CUDA unavailable
- Changed `hf3fs_iov_import_gpu()` to return `-ENOTSUP` instead of success with nullptr
- Added cache invalidation in `hf3fs_iovdestroy_gpu()`

### src/client/storage/StorageClient.h
- Added `isGpuMemory_` flag and `isGpuMemory()` accessor to `IOBuffer`
- Added constructor `IOBuffer(RDMABuf, bool isGpuMemory)`
- Added `registerGpuIOBuffer()` method declaration

### src/client/storage/StorageClient.cc
- Implemented `registerGpuIOBuffer()` method

### src/client/storage/StorageClientImpl.cc
- Modified `buildBatchRequest<ReadIO>` to track GPU buffers and disable inline data
- Modified read response handling to skip CPU memcpy/checksum for GPU buffers
- Modified write path to skip CPU checksum/inline for GPU buffers

### src/lib/common/GpuShm.h
- Removed unused `RDMABufGpu.h` include

### src/lib/common/GpuShm.cc
- Changed include from `RDMABufGpu.h` to `RDMABuf.h`
- Simplified `registerForIO()` to not use RDMABufGpu
- Updated `memh()` to create IOBuffer with `isGpuMemory=true`

Remaining Work
--------------
- **CUDA Runtime Integration**: The actual CUDA API calls are still placeholders.
  Real implementation requires:
  - `cudaGetDeviceCount()` / `cudaGetDeviceProperties()` for device detection
  - `cudaMalloc()` / `cudaFree()` for GPU memory allocation
  - `cudaIpcGetMemHandle()` / `cudaIpcOpenMemHandle()` for IPC
  - Optional: `cuMemExportToShareableHandle()` for dmabuf export

- **Testing**: Full integration testing on Linux with RDMA + CUDA hardware needed.

