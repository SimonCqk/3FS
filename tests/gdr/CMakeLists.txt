# GDR (GPU Direct RDMA) Tests
#
# All GDR test sources are consolidated here for unified build control.
# Unit tests use mock CUDA runtime (no real GPU required).
# Integration tests use GTEST_SKIP when hardware unavailable.
#
# Test sources reference files from tests/common/, tests/lib/, tests/fuse/
# via file(GLOB) below. MockCudaRuntime.h provides link-time CUDA stubs.

if(ENABLE_GDR OR HF3FS_GDR_ENABLED)
  # libfuse3 search paths (mirrored from src/fuse/CMakeLists.txt)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    link_directories(/usr/local/lib/x86_64-linux-gnu/ /usr/lib64 /usr/local/lib64)
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    link_directories(/usr/local/lib/aarch64-linux-gnu/ /usr/lib64 /usr/local/lib64)
  endif()

  # Collect all GDR test sources from their natural directories
  set(GDR_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/net/ib/TestAcceleratorMemoryMock.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/net/ib/TestRDMABufAccelerator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/api/TestUsrbIoGdr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../fuse/TestIovTableGdr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mocks/MockCudaRuntime.h
  )

  target_add_test(test_gdr hf3fs_api common hf3fs_fuse fdb mgmtd-fbs)
  target_sources(test_gdr PRIVATE ${GDR_TEST_SOURCES})
  target_compile_definitions(test_gdr PRIVATE HF3FS_GDR_ENABLED)
  target_include_directories(test_gdr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..  # For tests/gdr/mocks/ include path
    ${CMAKE_SOURCE_DIR}/src         # For source headers
  )
endif()
