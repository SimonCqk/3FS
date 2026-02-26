# Technical Proposal: GPU Direct RDMA (GDR) for 3FS usrbio

- **Document Type:** Technical Proposal
- **Status:** Draft
- **Authors:** 3FS Team
- **Target Branch:** `main`
- **Last Updated:** 2026-03-10

---

## 1. Motivation

In current implementation, when GPU workloads need to read or write data through 3FS, they must bounce through host buffers — the storage service transfers data to host RAM via RDMA, and the application then copies it to GPU memory (or vice versa for writes). This host bounce adds latency, consumes CPU memory bandwidth, and becomes a throughput bottleneck for large sequential reads and medium random reads common in model training and inference.

GPU Direct RDMA (GDR) eliminates this bounce by allowing read from and write to GPU VRAM directly over PCIe. With GDR, data moves on a single path — storage server → RDMA → GPU VRAM — with zero CPU-side copies.

However, integrating GDR into 3FS is not simply a matter of calling `ibv_reg_mr` on a CUDA pointer. Production deployments run the application and the fuse daemon as separate processes, so GPU memory must be shared across process boundaries via CUDA IPC handle export/import, and RDMA memory regions must be registered independently on both sides.

This proposal introduces GDR as a first-class capability, defining API contracts, registration flows, lifecycle ordering, and data paths for GPU memory through explicit `_device` API variants that complement the existing host API.

---

## 2. Goals and Non-Goals

### Goals

1. Extend usrbio with explicit `_device` API variants for GPU memory alongside the original host API, sharing lifecycle operations (`iovunlink`, `iovdestroy`) across both paths.
2. Define deterministic compile-time and runtime fallback behavior for CPU-only environments.
3. Specify the complete RDMA memory registration design for GPU buffers, including cross-process IPC import.
4. Specify strict lifecycle ordering for GPU resources to prevent stale rkeys, use-after-free, and resource leaks.
5. Support both single-process (app allocates + does I/O) and cross-process (app allocates, fuse daemon does I/O) deployment models.

### Non-Goals

1. Changes to the storage protocol, RPC schema, or chain replication logic.
2. Non-CUDA accelerator backends (AMD ROCm, Intel Level Zero, etc.).
3. Guaranteed performance parity with host path for all workload shapes.
4. Merging host and device API into a single overloaded entry point.

---

## 3. Design

### 3.1 API

This proposal extends the existing usrbio C API surface with explicit `_device` variants for GPU memory, keeping the original host API unchanged. GPU intent is expressed through separate entry points with a `device_id` parameter, rather than overloading the `numa` parameter:

```c
// Host memory: original API, unchanged
hf3fs_iovcreate(&iov, mount, size, block_size, /*numa=*/0);   // NUMA node 0
hf3fs_iovcreate(&iov, mount, size, block_size, /*numa=*/-1);  // no NUMA binding

// Device memory: explicit _device variants
hf3fs_iovcreate_device(&iov, mount, size, block_size, /*device_id=*/0);  // GPU 0
hf3fs_iovcreate_device(&iov, mount, size, block_size, /*device_id=*/1);  // GPU 1
```

The host lifecycle functions — `hf3fs_iovcreate`, `hf3fs_iovopen`, `hf3fs_iovwrap` — retain their original semantics: `numa >= 0` binds to a NUMA node, `numa < 0` means no NUMA binding. They do **not** accept GPU intent. The device API provides three parallel entry points:

- `hf3fs_iovcreate_device(iov, mount, size, block_size, device_id)` — always available; falls back to host memory if GDR is unavailable.
- `hf3fs_iovopen_device(iov, id, mount, size, block_size, device_id)` — only available when `HF3FS_GDR_ENABLED` is defined at compile time.
- `hf3fs_iovwrap_device(iov, device_ptr, id, mount, size, block_size, device_id)` — only available when `HF3FS_GDR_ENABLED` is defined at compile time.

The remaining lifecycle functions — `hf3fs_iovunlink`, `hf3fs_iovdestroy` — are shared and dispatch to the correct path based on the runtime type of the iov (host vs. GPU, detected via `kGpuIovMagicNuma`). Additional query functions (`hf3fs_iov_mem_type`, `hf3fs_iov_device_id`, `hf3fs_gdr_available`, `hf3fs_gdr_device_count`) let callers inspect the actual buffer type and runtime capability without conditional compilation.

**Fallback behavior.** Two independent gates govern whether the GPU path is actually taken. At compile time, `#ifdef HF3FS_GDR_ENABLED` strips all GPU code when OFF — `iovopen_device` and `iovwrap_device` are not even declared in the header. At runtime, `hf3fs_gdr_available()` means the GDR manager initialized successfully and a GPU memory-region cache is available. It is intentionally a coarse capability signal, not a guarantee that every subsequent GPU registration will succeed; lower-level failures such as verbs registration or driver/`nvidia_peermem` issues may still surface during buffer creation or first-use registration in the fuse daemon.

`hf3fs_iovcreate_device` is the only device API with implicit fallback — when GDR is unavailable (either compile-time or runtime), it silently allocates a host buffer at `numa=0`. This makes "prefer GPU if available" the zero-effort default. In contrast, `iovopen_device` and `iovwrap_device` are compile-time gated by `#ifdef HF3FS_GDR_ENABLED` and are absent entirely in non-GDR builds, because they operate on explicitly GPU-typed objects and silent fallback would be semantically incorrect. Callers targeting these APIs should guard usage with `#ifdef HF3FS_GDR_ENABLED` or check `hf3fs_gdr_available()` before calling.

---

## 4. Architecture

### 4.1 Process Model

3FS GPU I/O always involves two OS processes. The diagram below combines buffer publication, fuse-side import, and the RDMA data path so the control plane and data plane can be read together:

```
┌──────────────────────────── Application Process ────────────────────────────┐
│ [1] usrbio API (host + _device variants)                                   │
│     hf3fs_iovcreate/open/wrap + hf3fs_iovcreate/open/wrap_device           │
│     hf3fs_prep_io / hf3fs_submit_ios / hf3fs_wait_for_ios                  │
│                                                                             │
│     Host buffer branch                           GPU buffer branch          │
│     - /dev/shm/hf3fs-iov-*                       - cudaMalloc / wrapped ptr │
│     - iov->base = mmap ptr                       - iov->base = devicePtr    │
│     - iovh = ShmBuf*                             - iovh = GpuIovHandle*     │
│                                                  - ipcHandle + local MR     │
│                                                  - RegionCache by devicePtr │
│                                                                             │
│ [2] Publish / identify buffer                                               │
│     host: {uuid}[.b{block_size}]                                           │
│     gpu : {uuid}.gdr.d{N} -> gdr://v1/device/{N}/size/{S}/ipc/{hex}        │
│                                                                             │
│ [3] Submit I/O through host shared-memory ring -------------------------┐   │
└──────────────────────────────────────────────────────────────────────────┼───┘
                                                                           │
                      control plane + metadata discovery + I/O ring        │
                                                                           ▼
┌────────────────────── 3FS namespace and shared-memory bridge ──────────────┐
│ /mount/3fs-virt/iovs/*    lookup / reopen / publish visibility             │
│ POSIX SHM IoRing          submission + completion transport (host only)    │
└──────────────────────────────────────────────┬──────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────── Fuse Daemon Process ──────────────────────────┐
│ [4] Import / lookup                                                         │
│     - shmsById: host SHM reopened by second mmap                            │
│     - gpuShmsById: parse .gdr URI -> cudaIpcOpenMemHandle                   │
│     - daemon gets its own devicePtr and registers its own verbs MR          │
│                                                                             │
│ [5] Dispatch                                                                 │
│     lookupBufs() -> ShmBufForIO | GpuShmBufForIO                            │
│     IOBuffer -> RDMABufUnified(Host | Gpu)                                  │
│     StorageClient / IBSocket build requests without CPU deref on GPU ptr    │
│                                                                             │
│ [6] Completion                                                               │
│     CQE written back to the host ring; application observes completion      │
└──────────────────────────────────────────────┬──────────────────────────────┘
                                               │
                                               │ data plane: RDMA READ / WRITE
                                               ▼
                                    ┌──────────┴──────────┐
                                    │ HCA / NIC (IB verbs) │
                                    │ Host RAM or GPU VRAM │
                                    │ accessed over PCIe   │
                                    └──────────┬──────────┘
                                               │
                                               │ zero-copy GPU path uses
                                               │ nvidia_peermem + GPU MR
                                               ▼
                                     ┌────────────────────┐
                                     │ Storage Server      │
                                     │ disk / cache / RDMA │
                                     └────────────────────┘
```

The two processes communicate through:
1. **Filesystem namespace:** symlinks in `/mount/3fs-virt/iovs/` carry buffer metadata (UUID, device ID, IPC handle).
2. **CUDA IPC:** the 64-byte `cudaIpcMemHandle_t` is hex-encoded in the symlink target URI.
3. **Shared memory ring:** the I/O ring (`IoArgs`, `IoSqe`, `IoCqe`) lives in POSIX SHM for submission/completion.

### 4.2 Key Data Structures

#### Application side: iov → GpuIovHandle → RDMA registration

- `hf3fs_iov` remains the user-visible object for both host and GPU buffers. `iov->base` is either an `mmap` host pointer or a CUDA device pointer, and `iov->iovh` stays an opaque polymorphic handle.
- For GPU iovs, `iov->iovh` points to `GpuIovHandle`, which carries the local GPU state: `devicePtr`, `deviceId`, `size`, ownership flags, exported `cudaIpcMemHandle_t`, and the process-local RDMA registration handle.
- `GDRManager` owns runtime capability detection and GPU-to-IB affinity discovery. It is the entry point for deciding whether a GPU buffer can take the fast path.
- `AcceleratorMemoryRegionCache` caches registrations by `devicePtr`, so repeated operations on the same allocation reuse the same GPU MR rather than re-running `ibv_reg_mr`.
- Each `AcceleratorMemoryRegion` stores the GPU buffer descriptor plus per-IB-device `ibv_mr*` and `rkey` state used during RDMA request construction.

#### Fuse daemon side: GpuShmBuf → I/O dispatch → RDMA

- The daemon stores imported GPU buffers in `gpuShmsById`. Each `GpuShmBuf` contains the imported device pointer, `deviceId`, `size`, a daemon-local `AcceleratorMemoryRegion`, and lazily created `IOBuffer` handles.
- During request execution, `lookupBufs()` resolves a UUID to `GpuShmBufForIO`, which is just an offsetted view over the imported GPU allocation for one I/O operation.
- `GpuShmBufForIO::ptr()` returns a GPU device address, not a CPU-dereferenceable pointer. That constraint is why downstream abstractions must stay type-aware.
- `GpuShmBufForIO::memh()` produces an `IOBuffer` backed by `RDMABufUnified(Gpu)`, so `StorageClient` and `IBSocket` can operate on host and GPU buffers through one interface.
- `RDMABufAccelerator` is the final GPU RDMA wrapper. It provides the address, length, and `getMR()` access needed to build verbs requests without introducing host-side copies.

**Invariant:** `iov->iovh` is polymorphic — for host buffers it points to `ShmBuf*`, for GPU buffers it points to `GpuIovHandle*`. The runtime discriminant is `iov->numa == kGpuIovMagicNuma` combined with presence in the global `gGpuIovHandles` map.

---

## 5. Implementations

### 5.1 GDR Action Chain

The GDR data path touches the following 3FS components, in order:

1. `hf3fs_usrbio.h` declares explicit `_device` entry points alongside the original host API; `iovopen_device` and `iovwrap_device` are compile-time gated by `#ifdef HF3FS_GDR_ENABLED`.
2. `UsrbIo.cc` applies the compile-time and runtime gates: `#ifdef HF3FS_GDR_ENABLED` plus `hf3fs_gdr_available()`.
3. `UsrbIoGdr.cc` performs GPU-specific buffer creation or wrapping, exports `cudaIpcMemHandle_t`, and publishes `.gdr` metadata into the fuse namespace.
4. `AcceleratorMemory.h/.cc` owns GPU detection, GPU-to-IB affinity, MR caching, and per-IB-device `ibv_reg_mr` registration.
5. `IovTable.cc` in the fuse daemon reopens the published GPU buffer, parses the URI, imports the CUDA IPC handle, and creates a daemon-local MR.
6. `IoRing.cc` and `FuseClients.cc` resolve the UUID to `GpuShmBufForIO` and carry it through the host-resident submission/completion ring.
7. `RDMABufAccelerator` and `RDMABufUnified(Gpu)` provide the uniform RDMA abstraction consumed by `IBSocket` / `StorageClient`, which finally issue `ibv_post_send` for RDMA read or write.

When `HF3FS_GDR_ENABLED` is not defined at compile time, all GPU types (`GpuIovHandle`, `GpuShmBuf`, `RDMABufAccelerator`, etc.) are excluded. `IoBufForIO` becomes a plain `ShmBufForIO` instead of a variant, eliminating dispatch overhead entirely.

### 5.2 GPU IOV Metadata Contract

A GPU iov spans three metadata layers: the user-visible `hf3fs_iov`, the fuse namespace entry, and per-process internal state. That contract must be explicit because reopen/import depends on it.

**Namespace contract.**

- Host iov filename: `{uuid}[.b{block_size}]`
- GPU iov filename: `{uuid}.gdr.d{device_id}`
- GPU iov symlink target: `gdr://v1/device/{device_id}/size/{size}/ipc/{hex-encoded-64-byte-handle}`

For GPU iovs, the filename suffix `.d{N}` and the URI segment `device/{N}` must agree. The URI is the only cross-process source for the exported CUDA IPC handle and logical buffer size.

**User-visible `hf3fs_iov` field contract.**

- `iov->id`: 16-byte UUID used as the logical identity of the published buffer.
- `iov->base`: process-local pointer. For GPU iovs this is a CUDA device pointer in that process' CUDA address space; it is not CPU-dereferenceable.
- `iov->iovh`: opaque polymorphic handle. Host iovs point to `ShmBuf*`; GPU iovs point to `GpuIovHandle*`.
- `iov->numa`: for host iovs, reflects the NUMA binding. For GPU iovs, the implementation stores `kGpuIovMagicNuma` (`-0x6472`) as the runtime discriminant after successful create/open/wrap via `_device` APIs. This value is not a NUMA node — it is a magic sentinel used for type dispatch.
- `iov->size`: logical transfer span. For GPU reopen/import, the published namespace metadata is the authoritative source of size in the current implementation.
- `iov->block_size`: retained for API symmetry and future partitioning, but current GPU naming and `GpuShmBuf::memh()` use whole-buffer granularity. `block_size` is not encoded in the `.gdr` namespace entry and is not yet used to partition GPU memory handles.

**Open/reopen contract.**

- `hf3fs_iovcreate_device(iov, mount, size, block_size, device_id)` allocates new GPU memory and generates a new UUID. Falls back to host memory if GDR is unavailable.
- `hf3fs_iovwrap_device(iov, device_ptr, id, mount, size, block_size, device_id)` publishes caller-owned GPU memory under a caller-supplied UUID. Requires `HF3FS_GDR_ENABLED` at compile time.
- `hf3fs_iovopen_device(iov, id, mount, size, block_size, device_id)` imports an already published GPU iov by UUID. Requires `HF3FS_GDR_ENABLED` at compile time. The current implementation validates device identity from the `.gdr.d{N}` entry and uses URI size as the source of truth; callers should still pass matching `size` and `block_size` for forward compatibility and API clarity.
- `hf3fs_iorcreate*()` is orthogonal: the submission/completion ring remains host shared memory even when the data iov is GPU memory.

**Uniqueness and collision semantics.**

GPU publication requires UUID uniqueness within the mount namespace. A publish attempt that collides with an existing UUID should be treated as a metadata conflict, not as reopening the same buffer. The current implementation tolerates `symlink(...)=EEXIST` in the GPU path; tightening this to fail-fast is tracked as follow-up work.

### 5.3 Ownership and Lifetime

GPU GDR support introduces three distinct owners: the underlying GPU allocation owner, each process-local CUDA IPC import owner, and each process-local RDMA MR owner.

| API / role | Owns underlying VRAM | Owns imported CUDA mapping | Owns local RDMA MR | `iovdestroy` effect |
|-----------|----------------------|----------------------------|--------------------|---------------------|
| `hf3fs_iovcreate_device` | library in creator process | n/a until another process opens/imports | creator process | unpublish, invalidate MR, then `cudaFree` |
| `hf3fs_iovwrap_device` | caller/framework | n/a until another process opens/imports | wrapper process | unpublish and invalidate MR only; VRAM stays owned by caller |
| `hf3fs_iovopen_device` | exporter process | opener process | opener process | close imported mapping and invalidate opener-side MR |
| fuse daemon import | exporter process | fuse daemon | fuse daemon | close imported mapping and invalidate daemon-side MR |

The important lifetime rules are:

1. There is **no distributed refcount** across application processes and the fuse daemon.
2. `iovunlink` stops future discovery through the namespace, but it does **not** revoke already imported CUDA IPC mappings.
3. The exporter must keep the underlying GPU allocation alive until all outstanding I/O completes and all importer-side views have been destroyed.
4. For `iovwrap`, the external framework must not free or reuse the wrapped allocation until 3FS I/O and imports are fully torn down.
5. The safest shutdown order is: stop new submissions → wait for completions → destroy importer-side views → unlink/destroy the exporter.

### 5.4 Coherency Model

GPU GDR buffers are shared between CUDA execution engines and the RDMA NIC, not between CPU threads and the NIC. The coherency model therefore needs to be stated explicitly.

**What completion means.**

- `hf3fs_submit_ios()` only queues work.
- A successful CQE observed through `hf3fs_wait_for_ios()` means the storage-side RDMA operation is complete for that I/O.
- For a read, the target GPU memory range has been written by the NIC when completion is reported.
- For a write, the source GPU memory range has been consumed by the storage path when completion is reported.

**CPU visibility.**

Neither the application nor the fuse daemon may treat `iov->base` for a GPU iov as CPU-addressable memory. CPU `memcpy`, inline payload construction, checksum calculation over the pointer, and similar host operations are invalid on the GPU path.

**Explicit synchronization.**

`hf3fs_iovsync(iov, direction)` exists as a conservative fence API:

- `direction = 0`: before RDMA, make preceding GPU writes visible to the NIC.
- `direction = 1`: after RDMA, make NIC writes conservatively visible before dependent GPU work.

In the current implementation, both directions map to `cudaDeviceSynchronize()`. That gives a correct but coarse device-wide fence:

- It is conservative.
- It can serialize unrelated work on the same CUDA device.
- It is not stream-aware.
- It does not express partial-buffer or event-based ordering.

In many deployments `hf3fs_wait_for_ios()` is sufficient for post-read consumption because the NIC write into VRAM is already complete. `hf3fs_iovsync(..., 1)` should therefore be viewed as a conservative escape hatch, not as a mandatory step after every read.

### 5.5 CUDA IPC and Cross-Process Memory Sharing

The application process allocates GPU memory and must share it with the fuse daemon, which runs as a separate OS process. CUDA IPC provides the mechanism.

**Export (application side).** `cudaIpcGetMemHandle(&handle, devicePtr)` produces a 64-byte opaque handle that encodes enough information for another process on the same machine to map the same GPU allocation.

**Transport.** The IPC handle is hex-encoded (128 hex characters) and embedded in the fuse symlink target URI:

```
gdr://v1/device/0/size/1073741824/ipc/aabb...ff00
```

This piggybacks on the existing fuse namespace — no additional control-plane protocol is needed for the primary path.

**Import (fuse daemon side).** When `IovTable::addIov` encounters a `.gdr` key, it parses the URI, extracts the IPC handle bytes, and calls `cudaIpcOpenMemHandle(&importedPtr, cudaHandle, cudaIpcMemLazyEnablePeerAccess)`. The returned `importedPtr` is a valid device pointer in the daemon's CUDA address space, backed by the same physical GPU memory.

**Dual-side MR registration.** Both processes independently call `ibv_reg_mr` on their respective device pointers. This is required because IB memory regions are per-process. The daemon-side MR is the one used for actual storage RDMA operations; exporter-side and opener-side MRs are local registrations owned by those processes.

### 5.6 Read Path (Storage → GPU)

When the application submits a read I/O against a GPU iov, the fuse daemon dispatches it through the GPU data path:

1. The application prepares the read with `hf3fs_prep_io(...)` and submits it through the host-resident I/O ring.
2. The fuse daemon wakes up, processes the SQE, and resolves the target UUID through `lookupBufs(uuid)`.
3. For a GPU iov, the host-shm lookup misses and `gpuShmsById` returns a `GpuShmBufForIO` view over the imported GPU allocation.
4. The daemon derives a GPU device address through `ptr()` and a GPU-capable `IOBuffer` through `memh()`, which exposes the correct GPU MR and `rkey` state.
5. `StorageClient::read(...)` and the RDMA batch builder treat that buffer through `RDMABufUnified(Gpu)`, so no host dereference or CPU copy is required.
6. `ibv_post_send(RDMA READ)` causes the HCA to write file data directly into GPU VRAM over PCIe via `nvidia_peermem`.
7. After `hf3fs_wait_for_ios()` reports completion, the GPU memory range contains the requested file data.

The critical insight is unchanged: the fuse daemon orchestrates the operation but does not dereference the data. The HCA performs the PCIe transfer into VRAM directly.

### 5.7 Write Path (GPU → Storage)

The write path is symmetric to the read path. The application submits a write against a GPU iov, the fuse daemon resolves it to the same `GpuShmBufForIO` + `RDMABufUnified(Gpu)` abstraction, and issues an `ibv_post_send(RDMA WRITE)`. The HCA reads directly from GPU VRAM and transmits to the storage server with zero CPU-side copies.

The one extra requirement is producer ordering: if the GPU buffer was just written by CUDA kernels, the caller should ensure those writes are visible before submit. In the current API that conservative fence is `hf3fs_iovsync(&iov, 0)`.

### 5.8 StorageClient Changes for GPU Buffers

Adding GPU MRs alone is not enough. The StorageClient path had several implicit host-memory assumptions that had to be removed.

1. `IOBuffer` now wraps `RDMABufUnified(Host | Gpu)` rather than a host-only `RDMABuf`.
2. GPU-capable request builders use `subrangeRemote()` / `toRemoteBuf()` so they can operate on either host or GPU buffers without CPU dereference.
3. Inline read data is disabled when any request in the batch targets GPU memory; a CPU `memcpy` into a device pointer would be invalid.
4. Inline write payload construction is disabled for GPU buffers for the same reason.
5. Client-side CPU checksum verification is skipped for GPU read buffers.
6. Client-side CPU checksum generation is skipped for GPU write buffers; checksum responsibility remains on the storage side after RDMA access.
7. Host-only helper paths such as `IOBuffer::subrange()` are intentionally invalid for GPU buffers and must not be used on the GDR path.

These changes are important because they preserve the storage protocol while removing accidental CPU touches on device memory.

### 5.9 Engineering Considerations

**MR registration cost.** `ibv_reg_mr` on GPU memory is expensive — it pins GPU pages and sets up IOMMU / peer-memory mappings. Registration is therefore cached in `AcceleratorMemoryRegionCache`, keyed by `devicePtr`.

**`nvidia_peermem` and runtime dependency.** GPU memory registration via `ibv_reg_mr` depends on the `nvidia_peermem` kernel module. `GDRManager::init()` checks `/sys/module/nvidia_peermem` at startup and warns if absent, but the actual registration failure surfaces later during `ibv_reg_mr` calls. This is why `hf3fs_gdr_available()` should be treated as a coarse capability check — it confirms GDR infrastructure is initialized, not that every subsequent MR registration will succeed.

**`IBV_ACCESS_RELAXED_ORDERING`.** GPU MRs are registered with relaxed ordering in addition to standard RDMA read/write flags. This is part of the intended fast path for GPU Direct transfers.

**Lazy `memh()` materialization.** `GpuShmBuf::registerForIO()` marks the buffer as ready, but actual `IOBuffer` / `RDMABufAccelerator` objects are materialized lazily on first `memh()` use.

**IovTable slot management.** GPU iovs consume a descriptor slot in the shared `AtomicSharedPtrTable<ShmBuf>` but store `nullptr` there; the real GPU object lives in `gpuShmsById`. Removal must therefore call `dealloc()` directly rather than rely on host-shm removal behavior.

### 5.10 Operational Requirements

The v1 operational contract should be stated explicitly:

1. Build with `-DENABLE_GDR=ON`; otherwise `iovopen_device` and `iovwrap_device` are excluded, and `iovcreate_device` falls back to host allocation.
2. The application process and the fuse daemon must run on the same machine; CUDA IPC is same-host only.
3. Both processes must see a compatible CUDA device view. If `CUDA_VISIBLE_DEVICES` or container remapping is used, the device ordinal used in the `.gdr.d{N}` namespace must resolve consistently in both processes.
4. RDMA devices must be initialized and available before the GDR path can do useful work.
5. File descriptors used with `hf3fs_prep_io()` still must be registered via `hf3fs_reg_fd()`.
6. The I/O ring created by `hf3fs_iorcreate*()` remains host shared memory even for GPU data paths.
7. Callers wrapping externally allocated GPU memory should generate collision-free UUIDs and maintain the wrapped allocation for the full exported lifetime.

### 5.11 Known Limitations

The proposal should be explicit about what is in scope for the first merge and what remains follow-up work.

1. **GPU↔NIC affinity uses sysfs-based topology.** The GPU→IB mapping queries PCIe BDF and NUMA node from sysfs to score each GPU↔NIC pair by proximity (same PCIe switch > same domain > same NUMA > round-robin fallback). This covers the common multi-GPU/multi-NIC topologies. Future refinement: integrate NVML `nvmlDeviceGetTopologyCommonAncestor` for finer-grained PCIe switch distance when available.
2. **`dmabuf` is out of scope.** The v1 GDR path uses `nvidia_peermem` + CUDA IPC exclusively. `dmabuf` (`ibv_reg_dmabuf_mr` + CUDA VMM export) was evaluated but removed from this iteration — it requires VMM-based allocation, `cuMemExportToShareableHandle`, fd passing infrastructure, and kernel 5.12+, none of which are wired. It may be revisited for vendor-neutral GPU support (AMD ROCm, Intel Level Zero) in a future proposal.
3. **GPU `block_size` semantics are incomplete.** The current GPU path uses whole-buffer `memh()` granularity rather than block-partitioned registration and caching.
4. **Synchronization is conservative, not stream-aware.** `hf3fs_iovsync()` is device-wide today and does not integrate with CUDA events or streams.
5. **Cross-process lifetime coordination is best-effort.** There is no distributed refcount or lease mechanism for GPU publishes/imports, so exporter teardown must be coordinated by the caller.
6. **Orphan cleanup is not yet symmetric with host shm cleanup.** Host iovs benefit from existing dead-process cleanup paths; GPU namespace entries and importer-side state still need a more explicit orphan-management story.
7. **The device API uses explicit selection, not pointer autodetection.** Callers must use the `_device` variants with a `device_id` parameter to express GPU intent.

## 6. Failure Handling

A GPU iov should become externally visible only after its `.gdr` namespace entry is created successfully. Failures before publication must roll back to "as if never created".

| Failure point | Visibility state | Cleanup action |
|---------------|------------------|----------------|
| `cudaMalloc` fails | not published | return error; nothing to roll back |
| GPU MR registration fails | not published | free or close the local GPU allocation/import and return error |
| CUDA IPC export fails | not published | invalidate local MR state, release the local GPU allocation/import, and return error |
| GPU symlink publication fails | not published | full rollback; do not leave a partially constructed iov visible in the namespace |
| Fuse `readlink` / URI parse / `cudaIpcOpenMemHandle` fails | owner remains published; importer not created | importer returns error; exporter-side iov remains valid |
| Fuse daemon-side MR registration fails | importer not usable | close daemon-side IPC import and return error |

The destroy path uses best-effort sequential cleanup: unlink first, invalidate local MR state, then release the CUDA resource. If one cleanup step fails, log and continue so that later steps still run.

## 7. Use Cases

### A. Single-process inference (GPU destination buffer, host I/O ring)

```c
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>

#include "hf3fs_usrbio.h"

static const size_t kModelBytes = 1ULL << 30;

int fd = open("/mnt/3fs/models/model.bin", O_RDONLY);
hf3fs_reg_fd(fd, 0);

struct hf3fs_iov iov;
hf3fs_iovcreate_device(&iov, "/mnt/3fs", kModelBytes, 0, /*device_id=*/0);

struct hf3fs_ior ior;
/* The submission/completion ring is still host shared memory. */
hf3fs_iorcreate4(&ior, "/mnt/3fs", 64, true, 32, 5000, /*host numa=*/0, 0);

hf3fs_prep_io(&ior, &iov, true, iov.base, fd, 0, kModelBytes, NULL);
hf3fs_submit_ios(&ior);

struct hf3fs_cqe cqes[64];
int n = hf3fs_wait_for_ios(&ior, cqes, 64, 1, NULL);
if (n > 0 && cqes[0].result >= 0 && hf3fs_iov_mem_type(&iov) == HF3FS_MEM_DEVICE) {
  /* Optional conservative fence before launching dependent kernels. */
  hf3fs_iovsync(&iov, 1);
}

hf3fs_iordestroy(&ior);
hf3fs_iovdestroy(&iov);
hf3fs_dereg_fd(fd);
close(fd);
```

### B. PyTorch integration (wrap an existing tensor)

```c
/* PyTorch allocated this GPU memory. Requires HF3FS_GDR_ENABLED build. */
void *tensor_ptr = /* t.data_ptr() */;
size_t tensor_size = /* t.nbytes() */;
uint8_t tensor_uuid[16] = { /* caller-generated UUID bytes */ };

struct hf3fs_iov iov;
hf3fs_iovwrap_device(&iov,
                     tensor_ptr,
                     tensor_uuid,
                     "/mnt/3fs",
                     tensor_size,
                     0,
                     /*device_id=*/0);

int fd = open("/mnt/3fs/data/batch.bin", O_RDONLY);
hf3fs_reg_fd(fd, 0);

struct hf3fs_ior ior;
hf3fs_iorcreate4(&ior, "/mnt/3fs", 1, true, 0, 5000, /*host numa=*/0, 0);

hf3fs_prep_io(&ior, &iov, true, tensor_ptr, fd, 0, tensor_size, NULL);
hf3fs_submit_ios(&ior);

struct hf3fs_cqe cqe;
hf3fs_wait_for_ios(&ior, &cqe, 1, 1, NULL);

/* Optional conservative fence before the next dependent CUDA kernel. */
hf3fs_iovsync(&iov, 1);

hf3fs_iordestroy(&ior);
hf3fs_iovdestroy(&iov);  /* Releases 3FS metadata/MRs only; PyTorch still owns tensor_ptr. */
hf3fs_dereg_fd(fd);
close(fd);
```

### C. CPU-only node (transparent fallback on `iovcreate_device`)

```c
struct hf3fs_iov iov;
int rc = hf3fs_iovcreate_device(&iov, "/mnt/3fs", 1ULL << 30, 0, /*device_id=*/0);

if (rc == 0 && hf3fs_iov_mem_type(&iov) == HF3FS_MEM_HOST) {
  /* GDR was unavailable, so the library allocated host memory instead. */
}
```

### D. Cross-process GPU sharing

```c
/* Process A: allocate and publish GPU memory. */
static const size_t kBytes = 1ULL << 30;
struct hf3fs_iov iov_a;
hf3fs_iovcreate_device(&iov_a, "/mnt/3fs", kBytes, 0, /*device_id=*/0);

uint8_t shared_id[16];
memcpy(shared_id, iov_a.id, sizeof(shared_id));
/* Pass shared_id to Process B via any control channel. */

/* Process B: open the same published GPU buffer (requires HF3FS_GDR_ENABLED). */
struct hf3fs_iov iov_b;
hf3fs_iovopen_device(&iov_b, shared_id, "/mnt/3fs", kBytes, 0, /*device_id=*/0);

/* Process B now has its own CUDA IPC import of the same underlying VRAM. */

hf3fs_iovdestroy(&iov_b);  /* closes Process B's imported mapping */
hf3fs_iovdestroy(&iov_a);  /* Process A must destroy last, because it owns the VRAM */
```

This cross-process case assumes both processes resolve GPU device 0 the same way. In containerized setups, device ordinal remapping must be consistent between the application and the fuse daemon.

## 8. Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| **Overload `numa` parameter** (negative `numa` encodes `device_id`) | Overloads a single parameter with two unrelated semantics (NUMA binding vs. GPU selection). Makes `numa < 0` ambiguous — "no NUMA binding" vs. "GPU device". Harder to read at call sites. Explicit `_device` variants are clearer and type-safe. |
| **Hard-fail `iovcreate_device` when GDR unavailable** | Breaks portability. Applications that prefer GPU but tolerate host fallback would need explicit capability checks everywhere. |
| **Always bounce through host buffers** | Defeats the entire purpose of GDR. Adds latency and consumes CPU memory bandwidth. |
| **Defer cross-process support** | The fuse daemon is always a separate process in production. Single-process-only GDR would be unusable in real deployments. |
| **`dmabuf` (`ibv_reg_dmabuf_mr`) instead of `nvidia_peermem`** | Requires CUDA VMM allocation (`cuMemCreate` + `cuMemMap` + `cuMemExportToShareableHandle`), which changes the application's allocation model. Also requires fd passing infrastructure, kernel 5.12+, and RDMA-core ≥ 34. The `nvidia_peermem` path works with standard `cudaMalloc` and existing kernel/driver stacks. `dmabuf` may be revisited for vendor-neutral GPU support. |
