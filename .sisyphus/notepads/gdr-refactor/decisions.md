# Decisions - GDR Refactor

## Architectural Decisions

(No decisions recorded yet - will be populated as tasks complete)

## Decision: GDR Fallback Strategy in hf3fs_iovcreate()

### Context
When `hf3fs_iovcreate()` is called with negative numa values (device hints), the system needs to handle cases where GDR is unavailable (either not compiled or not available at runtime).

### Decision
Implement a two-tier fallback mechanism:
1. **Compile-time**: Use `#ifdef HF3FS_GDR_ENABLED` to conditionally compile GDR code
2. **Runtime**: Check `GDRManager::instance().isAvailable()` before using GDR path

### Rationale
- **Silent fallback**: Users don't need to change code when moving between GDR-enabled and CPU-only systems
- **No crashes**: Graceful degradation to host memory (numa = 0)
- **Observability**: DEBUG-level logging allows troubleshooting without noise in production
- **Consistency**: Follows existing patterns in GpuShm.cc and other GDR-aware components

### Implementation
- Modified: `src/lib/api/UsrbIo.cc`
- Added headers: `common/net/ib/AcceleratorMemory.h`, `lib/api/hf3fs_usrbio_gdr.h`
- Device encoding preserved: `device_id = -(numa + 1)`

