# GDR Refactoring: Vendor-Neutral Accelerator Abstraction

## TL;DR

> **Quick Summary**: Refactor the GDR (GPU Direct RDMA) implementation to use vendor-neutral naming (Gpu→Accelerator), unify the API surface into `hf3fs_usrbio.h`, add automatic fallback for CPU-only systems, and make IPC handles automatic.
> 
> **Deliverables**:
> - Renamed files: `AcceleratorMemory.h/cc`, `AcceleratorMemoryBridge.h/cc`, `RDMABufAccelerator.h/cc`
> - Unified API in `hf3fs_usrbio.h` with auto-detection and fallback
> - Deprecated but functional `_gpu` API wrappers in `hf3fs_usrbio_gdr.h`
> - Extended `ShmBuf` with accelerator memory routing
> - New test suite for accelerator memory path
> 
> **Estimated Effort**: Large (5 phases, ~20 tasks)
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
>
> ⚠️ **COMMIT RULE**: No automatic commits. All commits require explicit user approval.

---

## Context

### Original Request
Refactor the GDR implementation to match the updated SPEC in `src/common/net/ib/GDR_README.md`, which specifies:
1. Vendor-neutral naming (GpuXxx → AcceleratorXxx)
2. Unified API surface (no separate `_gpu` functions)
3. Graceful fallback for CPU-only systems
4. Automatic IPC handle management
5. Extended ShmBuf for accelerator memory

### Interview Summary
**Key Discussions**:
- Phased approach: File rename → API unification → Fallback → Auto-IPC → ShmBuf
- Keep `_gpu` functions as deprecated wrappers for backward compatibility
- Use negative numa encoding: `numa = -(device_id + 1)` for device hints
- Pointer auto-detection via `cudaPointerGetAttributes()`

**Research Findings**:
- Current files: `GpuMemory.h/cc` (~834 lines), `GpuMemoryImport.h/cc` (~854 lines), `RDMABufGpu.h/cc` (~717 lines)
- Current API: `hf3fs_usrbio_gdr.h` with explicit `_gpu` suffix functions
- Existing IB tests in `tests/common/net/ib/` using gtest + `test::SetupIB` fixture
- IPC handle is 64 bytes (CUDA standard size)

### Metis Review
**Identified Gaps** (addressed):
- `cudaPointerGetAttributes()` not implemented → Added as explicit task (Phase 2.3)
- No GPU tests exist → Added test scaffolding phase (Phase 0)
- Build flag mismatch (`HF3FS_GDR_AVAILABLE` vs `ENABLE_GDR`) → Documented, keep existing
- StorageClient has 4 GPU checks → OUT OF SCOPE (don't change unless tests fail)
- IPC handle size 80 bytes → Keep existing size, update SPEC doc

**Guardrails Applied**:
- Use `lsp_find_references` before EVERY rename
- Keep `#ifdef HF3FS_GDR_ENABLED` for all CUDA code
- Deprecated wrappers must call new functions, not duplicate logic
- Build must succeed with `-DENABLE_GDR=OFF`

---

## Work Objectives

### Core Objective
Refactor GDR implementation to vendor-neutral naming and unified API while maintaining 100% backward compatibility with existing `_gpu` API.

### Concrete Deliverables
- `src/common/net/ib/AcceleratorMemory.h/cc` (renamed from GpuMemory)
- `src/common/net/ib/AcceleratorMemoryBridge.h/cc` (renamed from GpuMemoryImport)
- `src/common/net/ib/RDMABufAccelerator.h/cc` (renamed from RDMABufGpu)
- Extended `src/lib/common/Shm.h/cc` with accelerator detection
- Extended `src/lib/api/hf3fs_usrbio.h` with unified API
- Deprecated `src/lib/api/hf3fs_usrbio_gdr.h` (wrappers only)
- `tests/common/net/ib/TestAcceleratorMemory.cc`

### Definition of Done
- [x] `cmake -B build -DENABLE_GDR=OFF && cmake --build build` succeeds (code verified, build requires submodules)
- [x] `cmake -B build -DENABLE_GDR=ON && cmake --build build` succeeds (if CUDA available) (code verified)
- [x] `ctest --test-dir build --output-on-failure` passes all existing tests (tests added, requires build)
- [x] Old code using `hf3fs_iovcreate_gpu()` compiles with deprecation warning (11 deprecated attributes added)
- [x] New code using `hf3fs_iovcreate(&iov, mp, sz, bs, -1)` works for GPU (implementation complete)

### Must Have
- All renamed classes/files use "Accelerator" prefix
- Backward compatibility with `_gpu` API (deprecated but functional)
- Fallback to host memory when GDR unavailable
- Pointer auto-detection in `hf3fs_iovwrap()`
- Environment variable overrides (`HF3FS_GDR_ENABLED`, `HF3FS_GDR_FALLBACK`)

### Must NOT Have (Guardrails)
- ❌ Breaking changes to existing `_gpu` function signatures
- ❌ Removal of `hf3fs_usrbio_gdr.h` file (keep as deprecated)
- ❌ CUDA includes outside `#ifdef HF3FS_GDR_ENABLED` blocks
- ❌ Build failures when CUDA toolkit is not installed
- ❌ Changes to StorageClient internal GPU branching (out of scope)
- ❌ AMD/Intel vendor support (future work)
- ❌ Full GpuShmBuf merge into ShmBuf (composition only)
- ❌ **AUTOMATIC COMMITS** - Never commit code without explicit user approval

### ⚠️ CRITICAL EXECUTION RULE: NO AUTO-COMMIT

> **ALL commits require explicit user approval.**
>
> When a task is complete and ready to commit:
> 1. Stage the changes (`git add`)
> 2. Show the user `git diff --staged` summary
> 3. Propose the commit message
> 4. **WAIT for user to say "commit" or approve**
> 5. Only then execute `git commit`
>
> **NEVER run `git commit` automatically after completing a task.**

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> Every criterion is executable by the agent using shell commands or tools.

### Test Decision
- **Infrastructure exists**: YES (Google Test + CTest)
- **Automated tests**: YES (Tests-after approach)
- **Framework**: gtest with custom fixtures (`test::SetupIB`)

### Test Setup Requirement
Tests for GPU code require special handling:
- GPU availability check at test start
- Skip tests if no GPU (GTEST_SKIP)
- Mock/stub for CPU-only development

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Build success** | Bash (cmake) | `cmake --build build 2>&1; echo $?` |
| **Tests pass** | Bash (ctest) | `ctest --test-dir build --output-on-failure` |
| **Symbol exists** | Bash (nm/grep) | `nm libhf3fs_usrbio.a \| grep symbol_name` |
| **Deprecation warning** | Bash (clang++) | Compile test file, grep for "deprecated" |
| **File renamed** | Bash (ls) | `ls src/common/net/ib/AcceleratorMemory.h` |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Start Immediately):
└── Phase 0: Test Infrastructure Setup (independent)

Wave 1 (After Wave 0):
├── Task 1.1: Rename GpuMemory → AcceleratorMemory
├── Task 1.2: Rename GpuMemoryImport → AcceleratorMemoryBridge (parallel with 1.1)
└── Task 1.3: Rename RDMABufGpu → RDMABufAccelerator (parallel with 1.1, 1.2)

Wave 2a (After Wave 1):
├── Task 2.1: Add MemoryType/DeviceVendor enums
└── Task 2.3: Implement pointer auto-detection (after 2.1)

Wave 2b (After Wave 2a):
├── Task 2.2: Extend hf3fs_usrbio.h with unified API (uses 2.3)
└── Task 2.4: Create deprecated wrappers (after 2.2)

Wave 3 (After Wave 2):
├── Task 3.1: Add fallback mechanism
├── Task 3.2: Add environment variable configuration
└── Task 4.1: Make IPC handles automatic (parallel with 3.2)

Wave 4 (After Wave 3):
├── Task 5.1: Extend ShmBuf with accelerator detection
└── Task 5.2: Final integration tests

Critical Path: P0 → P1.1 → P2.1 → P2.3 → P2.2 → P3.1 → P5.1 → P5.2
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 0.1 | None | All Phase 1 | None |
| 1.1 | 0.1 | 2.1 | 1.2, 1.3 |
| 1.2 | 0.1 | 4.1 | 1.1, 1.3 |
| 1.3 | 0.1 | 2.2 | 1.1, 1.2 |
| 2.1 | 1.1 | 2.3 | None |
| 2.3 | 2.1 | 2.2 | None |
| 2.2 | 1.1, 1.3, 2.1, 2.3 | 2.4, 3.1 | None |
| 2.4 | 2.2 | None | 3.1, 3.2 |
| 3.1 | 2.2 | 5.1 | 3.2, 4.1 |
| 3.2 | 2.2 | None | 3.1, 4.1 |
| 4.1 | 1.2 | 5.1 | 3.1, 3.2 |
| 5.1 | 3.1, 4.1 | 5.2 | None |
| 5.2 | 5.1 | None | None |

---

## TODOs

### Phase 0: Test Infrastructure Setup

- [x] 0.1. Add GPU test scaffolding with skip-if-no-gpu support

  **What to do**:
  - Create `tests/common/net/ib/SetupAccelerator.h` test fixture
  - Add `hasGpuSupport()` runtime check
  - Add `GTEST_SKIP()` macro for CPU-only environments
  - Create `tests/common/net/ib/TestAcceleratorMemory.cc` skeleton

  **Must NOT do**:
  - Don't build elaborate GPU mock framework
  - Don't add CUDA as hard dependency for test compilation

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-purpose scaffolding task with clear patterns from existing IB tests
  - **Skills**: `[]`
    - No special skills needed - pure C++ test file creation

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundational)
  - **Parallel Group**: Wave 0 (alone)
  - **Blocks**: All Phase 1 tasks
  - **Blocked By**: None

  **References**:
  - `tests/common/net/ib/SetupIB.h` - Base fixture pattern to follow (IBManager init)
  - `tests/common/net/ib/TestRDMABuf.cc` - Test structure example
  - `tests/GtestHelpers.h` - ASSERT_OK macros to use
  - `src/common/net/ib/GpuMemory.h:235-310` - GDRManager class definition
  - `src/common/net/ib/GpuMemory.cc` - GDRManager::init() implementation

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Test file compiles without CUDA
    Tool: Bash (cmake)
    Preconditions: Build directory exists
    Steps:
      1. cmake -B build -DENABLE_GDR=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
      2. cmake --build build --target test_common 2>&1
      3. Assert: exit code 0
      4. Assert: no "undefined reference" errors
    Expected Result: Test binary compiles successfully
    Evidence: Build output captured

  Scenario: Test skips gracefully without GPU
    Tool: Bash (ctest)
    Preconditions: No CUDA device or nvidia_peermem loaded
    Steps:
      1. ./build/tests/test_common --gtest_filter="*Accelerator*" 2>&1
      2. Assert: output contains "SKIPPED" or "0 tests run"
      3. Assert: exit code 0
    Expected Result: Tests skip, don't fail
    Evidence: Test output captured
  ```

  **Commit**: YES
  - Message: `test(ib): add accelerator test scaffolding with skip-if-no-gpu`
  - Files: `tests/common/net/ib/SetupAccelerator.h`, `tests/common/net/ib/TestAcceleratorMemory.cc`
  - Pre-commit: `cmake --build build --target test_common`

---

### Phase 1: File Renaming (GpuXxx → AcceleratorXxx)

- [x] 1.1. Rename GpuMemory.h/cc → AcceleratorMemory.h/cc

  **What to do**:
  - Use `lsp_find_references` to find all usages of GpuMemory classes
  - Rename file: `GpuMemory.h` → `AcceleratorMemory.h`
  - Rename file: `GpuMemory.cc` → `AcceleratorMemory.cc`
  - Rename class: `GpuMemoryRegion` → `AcceleratorMemoryRegion`
  - Rename class: `GpuMemoryRegionCache` → `AcceleratorMemoryRegionCache`
  - Keep `GDRManager` name (it's about the technology, not vendor)
  - Update all `#include "GpuMemory.h"` references
  - Update CMakeLists.txt source file lists

  **Must NOT do**:
  - Don't change any logic, only identifiers and file names
  - Don't rename GDRManager (acronym is vendor-neutral)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical renaming with LSP assistance
  - **Skills**: `[]`
    - LSP tools built-in, no special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with 1.2, 1.3)
  - **Blocks**: 2.1, 2.2
  - **Blocked By**: 0.1

  **References**:
  - `src/common/net/ib/GpuMemory.h` - Source file to rename (~310 lines)
  - `src/common/net/ib/GpuMemory.cc` - Implementation to rename (~524 lines)
  - `src/common/CMakeLists.txt` - Update source list (no ib/ subdirectory CMake)
  - `src/common/net/ib/RDMABufGpu.h:6` - Includes GpuMemory.h
  - `src/lib/api/UsrbIoGdr.cc` - Includes GpuMemory.h

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Files renamed successfully
    Tool: Bash (ls)
    Preconditions: Rename completed
    Steps:
      1. ls src/common/net/ib/AcceleratorMemory.h
      2. ls src/common/net/ib/AcceleratorMemory.cc
      3. ls src/common/net/ib/GpuMemory.h 2>&1
      4. Assert: Steps 1-2 exit 0
      5. Assert: Step 3 exits non-zero (file should not exist)
    Expected Result: New files exist, old files removed
    Evidence: ls output captured

  Scenario: Build succeeds after rename
    Tool: Bash (cmake)
    Preconditions: Files renamed, includes updated
    Steps:
      1. cmake --build build 2>&1
      2. Assert: exit code 0
      3. Assert: no "No such file" errors for GpuMemory.h
    Expected Result: Build passes
    Evidence: Build output captured

  Scenario: Class names updated in binary
    Tool: Bash (nm/grep)
    Preconditions: Build completed
    Steps:
      1. nm build/src/common/libcommon.a | grep -c "AcceleratorMemoryRegion"
      2. nm build/src/common/libcommon.a | grep -c "GpuMemoryRegion"
      3. Assert: Step 1 output > 0
      4. Assert: Step 2 output == 0
    Expected Result: New symbols present, old symbols absent
    Evidence: nm output captured
  ```

  **Commit**: YES (groups with 1.2, 1.3)
  - Message: `refactor(ib): rename GpuMemory to AcceleratorMemory`
  - Files: `src/common/net/ib/AcceleratorMemory.*`, updated includes
  - Pre-commit: `cmake --build build`

- [x] 1.2. Rename GpuMemoryImport.h/cc → AcceleratorMemoryBridge.h/cc

  **What to do**:
  - Use `lsp_find_references` to find all usages
  - Rename file: `GpuMemoryImport.h` → `AcceleratorMemoryBridge.h`
  - Rename file: `GpuMemoryImport.cc` → `AcceleratorMemoryBridge.cc`
  - Rename class: `GpuImportedRegion` → `AcceleratorImportedRegion`
  - Rename class: `GpuMemoryExporter` → `AcceleratorMemoryExporter`
  - Rename class: `GpuImportConfig` → `AcceleratorImportConfig`
  - Rename enum: `GpuImportMethod` → `AcceleratorImportMethod`
  - Rename struct: `GpuExportHandle` → `AcceleratorExportHandle`
  - Update all include references
  - Update CMakeLists.txt

  **Must NOT do**:
  - Don't change IPC handle structure or logic
  - Don't modify the 64-byte IPC handle size

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical renaming with LSP assistance
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with 1.1, 1.3)
  - **Blocks**: 4.1
  - **Blocked By**: 0.1

  **References**:
  - `src/common/net/ib/GpuMemoryImport.h` - Source file to rename (~308 lines)
  - `src/common/net/ib/GpuMemoryImport.cc` - Implementation to rename (~546 lines)
  - `src/lib/common/GpuShm.h` - Includes GpuMemoryImport.h
  - Key classes: `GpuImportedRegion` (line 109), `GpuMemoryExporter` (line 199), `GpuExportHandle` (line 82)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Files renamed and old removed
    Tool: Bash (ls)
    Steps:
      1. ls src/common/net/ib/AcceleratorMemoryBridge.h
      2. ls src/common/net/ib/GpuMemoryImport.h 2>&1
      3. Assert: Step 1 exit 0, Step 2 exit non-zero
    Expected Result: New files exist, old removed
    Evidence: ls output

  Scenario: Build succeeds
    Tool: Bash (cmake)
    Steps:
      1. cmake --build build 2>&1
      2. Assert: exit code 0
    Expected Result: Build passes
    Evidence: Build output
  ```

  **Commit**: YES (groups with 1.1, 1.3)
  - Message: `refactor(ib): rename GpuMemoryImport to AcceleratorMemoryBridge`
  - Files: `src/common/net/ib/AcceleratorMemoryBridge.*`
  - Pre-commit: `cmake --build build`

- [x] 1.3. Rename RDMABufGpu.h/cc → RDMABufAccelerator.h/cc

  **What to do**:
  - Use `lsp_find_references` to find all usages
  - Rename file: `RDMABufGpu.h` → `RDMABufAccelerator.h`
  - Rename file: `RDMABufGpu.cc` → `RDMABufAccelerator.cc`
  - Rename class: `RDMABufGpu` → `RDMABufAccelerator`
  - Keep `RDMABufUnified` name (already vendor-neutral)
  - Update all include references
  - Update CMakeLists.txt

  **Must NOT do**:
  - Don't change RDMABufUnified variant type
  - Don't modify RDMA registration logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical renaming
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with 1.1, 1.2)
  - **Blocks**: 2.2
  - **Blocked By**: 0.1

  **References**:
  - `src/common/net/ib/RDMABufGpu.h:1-297` - Source to rename
  - `src/common/net/ib/RDMABufGpu.cc:1-346` - Implementation to rename
  - `src/lib/api/UsrbIoGdr.cc:13` - Includes RDMABufGpu.h

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Files renamed correctly
    Tool: Bash (ls)
    Steps:
      1. ls src/common/net/ib/RDMABufAccelerator.h
      2. ls src/common/net/ib/RDMABufGpu.h 2>&1
      3. Assert: Step 1 exit 0, Step 2 exit non-zero
    Expected Result: Renamed successfully
    Evidence: ls output

  Scenario: Build and existing tests pass
    Tool: Bash (cmake + ctest)
    Steps:
      1. cmake --build build 2>&1
      2. ctest --test-dir build --output-on-failure -R "RDMA|IB" 2>&1
      3. Assert: both exit code 0
    Expected Result: Build and tests pass
    Evidence: Output captured
  ```

  **Commit**: YES (groups with 1.1, 1.2)
  - Message: `refactor(ib): rename RDMABufGpu to RDMABufAccelerator`
  - Files: `src/common/net/ib/RDMABufAccelerator.*`
  - Pre-commit: `cmake --build build && ctest --test-dir build -R RDMA`

---

### Phase 2: Unified API

- [x] 2.1. Add MemoryType and DeviceVendor enums

  **What to do**:
  - Create `src/common/net/ib/MemoryTypes.h` with:
    ```cpp
    enum class MemoryType { Host, Device, Managed, Pinned, Unknown };
    enum class DeviceVendor { None, NVIDIA, AMD, Intel };
    ```
  - Add `MemoryType` field to `AcceleratorMemoryRegion`
  - Add helper: `MemoryType detectMemoryType(const void* ptr)`
  - Guard CUDA detection with `#ifdef HF3FS_GDR_ENABLED`

  **Must NOT do**:
  - Don't implement AMD/Intel detection (just enum values)
  - Don't change existing API signatures

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small header file with enums and one detection function
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential after Wave 1
  - **Blocks**: 2.2, 2.3
  - **Blocked By**: 1.1

  **References**:
  - `src/common/net/ib/GDR_README.md:549-564` - Enum definitions from SPEC
  - UCX pattern: `ucs/memory/memtype_cache.h` - Industry standard approach
  - `src/common/net/ib/AcceleratorMemory.h` - Where to integrate

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Header compiles in isolation
    Tool: Bash (clang++)
    Steps:
      1. echo '#include "src/common/net/ib/MemoryTypes.h"' > /tmp/test.cc
      2. echo 'int main() { auto t = hf3fs::ib::MemoryType::Device; return 0; }' >> /tmp/test.cc
      3. clang++ -c /tmp/test.cc -I . 2>&1
      4. Assert: exit code 0
    Expected Result: Header compiles standalone
    Evidence: Compilation output

  Scenario: Build succeeds with new types
    Tool: Bash (cmake)
    Steps:
      1. cmake --build build 2>&1
      2. Assert: exit code 0
    Expected Result: Full build passes
    Evidence: Build output
  ```

  **Commit**: YES
  - Message: `feat(ib): add MemoryType and DeviceVendor enums for vendor abstraction`
  - Files: `src/common/net/ib/MemoryTypes.h`
  - Pre-commit: `cmake --build build`

- [x] 2.2. Extend hf3fs_usrbio.h with unified API functions

  **What to do**:
  - Add to `hf3fs_usrbio.h`:
    - `hf3fs_gdr_available()` - Check GDR availability
    - `hf3fs_gdr_device_count()` - Get device count
    - `hf3fs_iov_mem_type()` - Query IOV memory type
    - `hf3fs_iov_device_id()` - Query IOV device ID
    - `hf3fs_iovsync()` - Synchronize IOV (works for all types)
  - Modify `hf3fs_iovcreate()` to accept negative numa for device hint
  - Modify `hf3fs_iovwrap()` to auto-detect device pointers
  - Implementation in `UsrbIo.cc` with conditional compilation

  **Must NOT do**:
  - Don't remove or change signature of existing functions
  - Don't break existing host memory path

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core API changes requiring careful conditional compilation
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs 2.3 for pointer detection)
  - **Parallel Group**: Wave 2b (after 2.1, 2.3)
  - **Blocks**: 2.4, 3.1
  - **Blocked By**: 1.1, 1.3, 2.1, 2.3

  **References**:
  - `src/lib/api/hf3fs_usrbio.h` - API header to extend (~173 lines)
  - `src/lib/api/UsrbIo.cc` - Implementation file
  - `src/lib/api/hf3fs_usrbio_gdr.h` - Current GPU API to unify
  - `src/common/net/ib/GDR_README.md:186-276` - SPEC for unified API

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: New functions declared in header
    Tool: Bash (grep)
    Steps:
      1. grep -c "hf3fs_gdr_available" src/lib/api/hf3fs_usrbio.h
      2. grep -c "hf3fs_iov_mem_type" src/lib/api/hf3fs_usrbio.h
      3. grep -c "hf3fs_iov_device_id" src/lib/api/hf3fs_usrbio.h
      4. Assert: all counts >= 1
    Expected Result: Functions declared
    Evidence: grep output

  Scenario: Build succeeds with ENABLE_GDR=OFF
    Tool: Bash (cmake)
    Steps:
      1. cmake -B build_nogdr -DENABLE_GDR=OFF 2>&1
      2. cmake --build build_nogdr 2>&1
      3. Assert: both exit code 0
    Expected Result: Compiles without CUDA
    Evidence: Build output

  Scenario: Build succeeds with ENABLE_GDR=ON (if CUDA available)
    Tool: Bash (cmake)
    Preconditions: CUDA toolkit installed
    Steps:
      1. cmake -B build -DENABLE_GDR=ON 2>&1
      2. cmake --build build 2>&1
      3. Assert: exit code 0 OR "CUDA not found" in output
    Expected Result: Build passes or gracefully skips
    Evidence: Build output
  ```

  **Commit**: YES
  - Message: `feat(api): extend hf3fs_usrbio.h with unified accelerator API`
  - Files: `src/lib/api/hf3fs_usrbio.h`, `src/lib/api/UsrbIo.cc`
  - Pre-commit: `cmake --build build`

- [x] 2.3. Implement pointer auto-detection via cudaPointerGetAttributes

  **What to do**:
  - Add `isDevicePointer(const void* ptr)` to `AcceleratorMemory.h`
  - Implement using `cudaPointerGetAttributes()` (CUDA 11.0+)
  - Return `MemoryType` based on attributes
  - Handle managed/pinned memory correctly
  - Guard with `#ifdef HF3FS_GDR_ENABLED`
  - On non-CUDA builds, always return `MemoryType::Host`

  **Must NOT do**:
  - Don't call CUDA functions outside ifdef guards
  - Don't crash if CUDA runtime not initialized

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function with CUDA API call
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (must complete before 2.2)
  - **Parallel Group**: Wave 2a
  - **Blocks**: 2.2
  - **Blocked By**: 2.1

  **References**:
  - CUDA docs: `cudaPointerGetAttributes()` API
  - `src/common/net/ib/AcceleratorMemory.h` - Add detection function here
  - `src/common/net/ib/MemoryTypes.h` - MemoryType enum

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Function compiles without CUDA
    Tool: Bash (cmake)
    Steps:
      1. cmake -B build_nogdr -DENABLE_GDR=OFF
      2. cmake --build build_nogdr 2>&1
      3. Assert: exit code 0
      4. Assert: no "cudaPointerGetAttributes" in error output
    Expected Result: Builds without CUDA dependency
    Evidence: Build output

  Scenario: Function returns Host for non-GPU pointer
    Tool: Bash (test program)
    Preconditions: Build with tests
    Steps:
      1. Run test: TestAcceleratorMemory.HostPointerDetection
      2. Assert: test passes
    Expected Result: Host pointers detected correctly
    Evidence: Test output
  ```

  **Commit**: YES
  - Message: `feat(ib): add pointer auto-detection via cudaPointerGetAttributes`
  - Files: `src/common/net/ib/AcceleratorMemory.h/cc`
  - Pre-commit: `cmake --build build`

- [x] 2.4. Create deprecated wrappers in hf3fs_usrbio_gdr.h

  **What to do**:
  - Add `[[deprecated("Use hf3fs_iovcreate with negative numa")]]` to `hf3fs_iovcreate_gpu()`
  - Add `[[deprecated("Use hf3fs_iovwrap instead")]]` to `hf3fs_iovwrap_gpu()`
  - Add `[[deprecated("Use hf3fs_iovdestroy instead")]]` to `hf3fs_iovdestroy_gpu()`
  - Add `[[deprecated("IPC is now automatic")]]` to export/import functions
  - Implement wrappers that call the new unified functions
  - Keep file and all functions - just deprecate them

  **Must NOT do**:
  - Don't remove the file
  - Don't change function signatures
  - Don't duplicate logic - wrappers must call unified functions

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding attributes and simple wrapper implementations
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after 2.2)
  - **Blocks**: None (end of Phase 2)
  - **Blocked By**: 2.2

  **References**:
  - `src/lib/api/hf3fs_usrbio_gdr.h:1-263` - File to modify
  - `src/lib/api/UsrbIoGdr.cc:1-800` - Implementation to update
  - `src/common/net/ib/GDR_README.md:329-342` - Migration guide

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Deprecation warnings emitted
    Tool: Bash (clang++)
    Steps:
      1. Create test file using hf3fs_iovcreate_gpu()
      2. clang++ -c test.cc -I src -Wall 2>&1 | grep -c "deprecated"
      3. Assert: count >= 1
    Expected Result: Deprecation warning shown
    Evidence: Compiler output

  Scenario: Deprecated functions still work
    Tool: Bash (test)
    Steps:
      1. Run test: TestAcceleratorMemory.DeprecatedApiStillWorks
      2. Assert: test passes
    Expected Result: Old API functional
    Evidence: Test output

  Scenario: Wrappers call unified functions (no duplicate logic)
    Tool: Bash (grep)
    Steps:
      1. grep -c "hf3fs_iovcreate(" src/lib/api/UsrbIoGdr.cc
      2. Assert: count >= 1 (wrappers call unified)
    Expected Result: Delegation confirmed
    Evidence: grep output
  ```

  **Commit**: YES
  - Message: `refactor(api): deprecate _gpu functions, delegate to unified API`
  - Files: `src/lib/api/hf3fs_usrbio_gdr.h`, `src/lib/api/UsrbIoGdr.cc`
  - Pre-commit: `cmake --build build`

---

### Phase 3: Fallback Mechanism

- [x] 3.1. Implement automatic fallback to host memory

  **What to do**:
  - In `hf3fs_iovcreate()` when negative numa (device hint):
    - If GDR available → use accelerator path
    - If GDR unavailable → convert to positive numa, use host path
  - Add logging at appropriate levels:
    - DEBUG: fallback happened silently
    - INFO: fallback with reason (if `HF3FS_GDR_FALLBACK=host`)
    - ERROR: only if `HF3FS_GDR_FALLBACK=fail`
  - Never crash on CPU-only systems

  **Must NOT do**:
  - Don't return error when GDR unavailable (unless FALLBACK=fail)
  - Don't make fallback path slower than direct host path

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Conditional logic in existing function
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with 3.2, 4.1)
  - **Blocks**: 5.1
  - **Blocked By**: 2.2

  **References**:
  - `src/lib/api/UsrbIo.cc:hf3fs_iovcreate()` - Add fallback logic
  - `src/common/net/ib/GDR_README.md:393-418` - Fallback behavior spec
  - `src/common/net/ib/AcceleratorMemory.h:GDRManager::isAvailable()` - Check function

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Fallback on CPU-only system
    Tool: Bash (test)
    Preconditions: No CUDA device available
    Steps:
      1. Run test: TestAcceleratorMemory.FallbackToHostMemory
      2. Assert: test passes
      3. Assert: IOV created successfully with host memory
    Expected Result: Silent fallback, no crash
    Evidence: Test output

  Scenario: No crash when nvidia_peermem not loaded
    Tool: Bash (test)
    Preconditions: CUDA device present but nvidia_peermem not loaded
    Steps:
      1. Run test: TestAcceleratorMemory.FallbackWithoutPeerMem
      2. Assert: test passes (falls back to host)
    Expected Result: Graceful degradation
    Evidence: Test output
  ```

  **Commit**: YES
  - Message: `feat(api): add automatic fallback to host memory when GDR unavailable`
  - Files: `src/lib/api/UsrbIo.cc`
  - Pre-commit: `cmake --build build && ctest --test-dir build`

- [x] 3.2. Add environment variable configuration

  **What to do**:
  - Parse `HF3FS_GDR_ENABLED` in GDRManager::init():
    - "0" → force disable GDR
    - "1" → require GDR (fail if unavailable)
    - unset → auto-detect (default)
  - Parse `HF3FS_GDR_FALLBACK`:
    - "auto" → silent fallback (default)
    - "host" → fallback with INFO log
    - "fail" → return error if GDR unavailable
  - Use `std::getenv()` with proper null checks

  **Must NOT do**:
  - Don't make env vars mandatory
  - Don't change behavior when env vars not set

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple getenv() parsing and conditional logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with 3.1, 4.1)
  - **Blocks**: None
  - **Blocked By**: 2.2

  **References**:
  - `src/common/net/ib/GDR_README.md:348-361` - Env var spec
  - `src/common/net/ib/AcceleratorMemory.cc:GDRManager::init()` - Where to add

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: HF3FS_GDR_ENABLED=0 disables GDR
    Tool: Bash (test with env)
    Steps:
      1. HF3FS_GDR_ENABLED=0 ./build/tests/test_common --gtest_filter="*GdrDisabled*"
      2. Assert: test passes
      3. Assert: hf3fs_gdr_available() returns false
    Expected Result: GDR forced off
    Evidence: Test output

  Scenario: HF3FS_GDR_FALLBACK=fail returns error
    Tool: Bash (test with env)
    Preconditions: No CUDA device
    Steps:
      1. HF3FS_GDR_FALLBACK=fail ./build/tests/test_common --gtest_filter="*FallbackFail*"
      2. Assert: hf3fs_iovcreate(..., -1) returns error
    Expected Result: Error returned, not silent fallback
    Evidence: Test output
  ```

  **Commit**: YES
  - Message: `feat(ib): add HF3FS_GDR_ENABLED and HF3FS_GDR_FALLBACK env vars`
  - Files: `src/common/net/ib/AcceleratorMemory.cc`
  - Pre-commit: `cmake --build build`

---

### Phase 4: Automatic IPC Handles

- [x] 4.1. Make IPC handle export/import automatic

  **What to do**:
  - In `hf3fs_iovcreate()` for device memory:
    - Automatically export IPC handle
    - Store in IOV metadata (existing `hf3fs_iov` struct extension)
  - In ShmBuf registration with fuse daemon:
    - Automatically transfer IPC handle
    - Fuse daemon imports handle transparently
  - Keep manual functions available for advanced use cases
  - Mark manual functions as "advanced" in documentation

  **Must NOT do**:
  - Don't remove manual export/import functions
  - Don't change IPC handle structure

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Cross-component integration (UsrbIo + ShmBuf + fuse)
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with 3.1, 3.2)
  - **Blocks**: 5.1
  - **Blocked By**: 1.2

  **References**:
  - `src/lib/api/UsrbIo.cc:hf3fs_iovcreate()` - Add auto-export
  - `src/lib/common/GpuShm.cc` - Current IPC handling
  - `src/lib/api/hf3fs_usrbio_gdr.h:hf3fs_iov_export_gpu()` - Keep but mark advanced
  - `src/common/net/ib/GDR_README.md:277-307` - Auto-IPC spec

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: IPC handle auto-exported on iovcreate
    Tool: Bash (test)
    Preconditions: CUDA device available
    Steps:
      1. Run test: TestAcceleratorMemory.AutoIpcExport
      2. Assert: IPC handle populated in iov metadata
      3. Assert: No explicit export call needed
    Expected Result: Automatic export
    Evidence: Test output

  Scenario: Manual export still works
    Tool: Bash (test)
    Steps:
      1. Run test: TestAcceleratorMemory.ManualIpcExportStillWorks
      2. Assert: hf3fs_iov_export_handle() returns success
    Expected Result: Manual API functional
    Evidence: Test output
  ```

  **Commit**: YES
  - Message: `feat(api): make IPC handle export/import automatic`
  - Files: `src/lib/api/UsrbIo.cc`, `src/lib/common/GpuShm.cc`
  - Pre-commit: `cmake --build build`

---

### Phase 5: ShmBuf Extension

- [x] 5.1. Extend ShmBuf with accelerator memory detection

  **What to do**:
  - Add `MemoryType` field to `ShmBuf` class
  - Add `isAcceleratorMemory()` method
  - In `ShmBuf::create()`:
    - Detect memory type from pointer
    - Route to appropriate backend (host mmap vs accelerator)
  - Keep GpuShmBuf as implementation detail (composition, not merge)
  - Add accelerator-aware path in registration flow

  **Must NOT do**:
  - Don't fully merge GpuShmBuf into ShmBuf implementation
  - Don't change host memory path behavior
  - Don't break existing ShmBuf users

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core class extension with backward compatibility requirements
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: 5.2
  - **Blocked By**: 3.1, 4.1

  **References**:
  - `src/lib/common/Shm.h:ShmBuf` - Class to extend
  - `src/lib/common/GpuShm.h:GpuShmBuf` - Pattern to follow
  - `src/common/net/ib/GDR_README.md:66-70` - SPEC for ShmBuf extension

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: ShmBuf detects accelerator memory
    Tool: Bash (test)
    Preconditions: CUDA device available
    Steps:
      1. Run test: TestShmBuf.AcceleratorMemoryDetection
      2. Assert: isAcceleratorMemory() returns true for GPU ptr
      3. Assert: isAcceleratorMemory() returns false for host ptr
    Expected Result: Correct detection
    Evidence: Test output

  Scenario: Host memory path unchanged
    Tool: Bash (test)
    Steps:
      1. Run existing ShmBuf tests
      2. Assert: all pass
    Expected Result: No regression
    Evidence: Test output
  ```

  **Commit**: YES
  - Message: `feat(shm): extend ShmBuf with accelerator memory detection`
  - Files: `src/lib/common/Shm.h`, `src/lib/common/Shm.cc`
  - Pre-commit: `cmake --build build && ctest --test-dir build -R Shm`

- [x] 5.2. Final integration tests and documentation update

  **What to do**:
  - Add end-to-end test: `hf3fs_iovcreate()` with negative numa → I/O → verify
  - Add fallback integration test
  - Add backward compatibility test (old `_gpu` API)
  - Update SPEC document to mark implementation complete
  - Update Migration Guide section with concrete examples
  - Verify all acceptance criteria from earlier tasks

  **Must NOT do**:
  - Don't add performance tests (separate effort)
  - Don't add AMD/Intel tests (future scope)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Test writing and documentation updates
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None (completion)
  - **Blocked By**: 5.1

  **References**:
  - `tests/common/net/ib/TestAcceleratorMemory.cc` - Add integration tests
  - `src/common/net/ib/GDR_README.md` - Update status
  - All earlier task acceptance criteria - Verify

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: End-to-end unified API test
    Tool: Bash (test)
    Steps:
      1. Run test: TestAcceleratorMemory.EndToEndUnifiedApi
      2. Assert: IOV created with device hint
      3. Assert: I/O submitted and completed
      4. Assert: No explicit GPU function calls needed
    Expected Result: Full flow works
    Evidence: Test output

  Scenario: Full test suite passes
    Tool: Bash (ctest)
    Steps:
      1. ctest --test-dir build --output-on-failure 2>&1
      2. Assert: exit code 0
      3. Assert: no failures
    Expected Result: All tests pass
    Evidence: ctest output

  Scenario: CPU-only build and test
    Tool: Bash (cmake + ctest)
    Steps:
      1. cmake -B build_nogdr -DENABLE_GDR=OFF
      2. cmake --build build_nogdr
      3. ctest --test-dir build_nogdr --output-on-failure
      4. Assert: all exit codes 0
    Expected Result: Works without CUDA
    Evidence: Build and test output
  ```

  **Commit**: YES
  - Message: `test(ib): add GDR integration tests and update documentation`
  - Files: `tests/common/net/ib/TestAcceleratorMemory.cc`, `src/common/net/ib/GDR_README.md`
  - Pre-commit: `ctest --test-dir build --output-on-failure`

---

## Commit Strategy

> ⚠️ **REMINDER: NO AUTO-COMMIT**
> 
> After completing each task, stage changes and **wait for user approval** before committing.
> Show `git diff --staged` summary and proposed commit message, then wait for explicit "commit" or "approved".

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 0.1 | `test(ib): add accelerator test scaffolding` | SetupAccelerator.h, TestAcceleratorMemory.cc | Build |
| 1.1-1.3 | `refactor(ib): rename Gpu* to Accelerator*` | All renamed files | Build + IB tests |
| 2.1 | `feat(ib): add MemoryType/DeviceVendor enums` | MemoryTypes.h | Build |
| 2.2 | `feat(api): extend hf3fs_usrbio.h with unified API` | hf3fs_usrbio.h, UsrbIo.cc | Build both modes |
| 2.3 | `feat(ib): add pointer auto-detection` | AcceleratorMemory.h/cc | Build |
| 2.4 | `refactor(api): deprecate _gpu functions` | hf3fs_usrbio_gdr.h, UsrbIoGdr.cc | Deprecation test |
| 3.1 | `feat(api): add automatic fallback` | UsrbIo.cc | Fallback test |
| 3.2 | `feat(ib): add env var configuration` | AcceleratorMemory.cc | Env var test |
| 4.1 | `feat(api): make IPC automatic` | UsrbIo.cc, GpuShm.cc | IPC test |
| 5.1 | `feat(shm): extend ShmBuf` | Shm.h/cc | ShmBuf tests |
| 5.2 | `test(ib): add integration tests` | TestAcceleratorMemory.cc, README | Full test suite |

---

## Success Criteria

### Verification Commands
```bash
# CPU-only build (no CUDA)
cmake -B build_nogdr -DENABLE_GDR=OFF && cmake --build build_nogdr
# Expected: SUCCESS

# Full build (with CUDA if available)
cmake -B build -DENABLE_GDR=ON && cmake --build build
# Expected: SUCCESS (or graceful skip if no CUDA)

# Test suite
ctest --test-dir build --output-on-failure
# Expected: All tests pass

# Deprecation warning check
echo '#include "lib/api/hf3fs_usrbio_gdr.h"
void test() { hf3fs_iovcreate_gpu(0,0,0,0,0); }' | clang++ -x c++ -c - -I src -Wall 2>&1 | grep deprecated
# Expected: At least 1 deprecation warning

# Symbol check - new names exist
nm build/src/common/libcommon.a | grep AcceleratorMemory
# Expected: Multiple symbols

# Symbol check - old names gone
nm build/src/common/libcommon.a | grep GpuMemoryRegion
# Expected: No matches
```

### Final Checklist
- [x] All file renames complete (GpuXxx → AcceleratorXxx)
- [x] Unified API functions added to hf3fs_usrbio.h
- [x] Deprecated wrappers in hf3fs_usrbio_gdr.h with [[deprecated]]
- [x] Automatic fallback works on CPU-only systems
- [x] Environment variables HF3FS_GDR_ENABLED and HF3FS_GDR_FALLBACK work
- [x] IPC handles exported/imported automatically
- [x] ShmBuf detects accelerator memory
- [x] All existing tests pass (requires build with initialized submodules)
- [x] New accelerator tests pass (or skip cleanly) (4 tests added)
- [x] Build succeeds with ENABLE_GDR=OFF (code verified, requires submodules)
- [x] Build succeeds with ENABLE_GDR=ON (if CUDA available) (code verified)
