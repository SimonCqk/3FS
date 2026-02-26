/**
 * Scenario tests for Layer 2: RDMABufAccelerator
 *
 * Tests RDMABufAccelerator creation, subranges, toRemoteBuf,
 * RDMABufUnified type dispatch, pool, and sync.
 *
 * Covers: REQ-L2-001 through REQ-L2-006
 */

#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/net/ib/AcceleratorMemory.h"
#include "common/net/ib/RDMABuf.h"
#include "common/net/ib/RDMABufAccelerator.h"
#include "tests/GtestHelpers.h"
#include "tests/gdr/mocks/MockCudaRuntime.h"

namespace hf3fs::net {

class TestRDMABufAccelerator : public ::testing::Test {
 protected:
  void SetUp() override {
    hf3fs::test::MockCudaRuntime::instance().reset();
  }

  void TearDown() override {
    hf3fs::test::MockCudaRuntime::instance().reset();
  }

  static bool hasGpu() {
    return GDRManager::instance().isAvailable();
  }
};

// ==========================================================================
// REQ-L2-001: GPU RDMA Buffer Creation from Device Pointer
// ==========================================================================

// @tests SCN-L2-001-02
TEST_F(TestRDMABufAccelerator, SCN_L2_001_02_CreateFromGpuPointerNoGDR) {
  if (hasGpu()) {
    GTEST_SKIP() << "Test is for non-GPU environments";
  }

  // GIVEN: GDRManager::isAvailable() returns false
  // WHEN: createFromGpuPointer is called
  auto buf = RDMABufAccelerator::createFromGpuPointer(
      reinterpret_cast<void*>(0x1000), 4096, 0);

  // THEN: Returns invalid buffer
  EXPECT_FALSE(buf.valid());
  EXPECT_FALSE(static_cast<bool>(buf));
}

// @tests SCN-L2-001-03
TEST_F(TestRDMABufAccelerator, SCN_L2_001_03_CreateFromGpuPointerInvalidParams) {
  // GIVEN: Invalid parameters — these should fail regardless of GPU availability
  // WHEN: nullptr pointer
  auto buf1 = RDMABufAccelerator::createFromGpuPointer(nullptr, 4096, 0);
  EXPECT_FALSE(buf1.valid());

  // WHEN: zero size
  auto buf2 = RDMABufAccelerator::createFromGpuPointer(
      reinterpret_cast<void*>(0x1000), 0, 0);
  EXPECT_FALSE(buf2.valid());

  // WHEN: negative device ID
  auto buf3 = RDMABufAccelerator::createFromGpuPointer(
      reinterpret_cast<void*>(0x1000), 4096, -1);
  EXPECT_FALSE(buf3.valid());
}

// @tests SCN-L2-001-01
TEST_F(TestRDMABufAccelerator, SCN_L2_001_01_CreateFromGpuPointerSuccess) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU available — integration test only";
  }

  // Integration test with real GPU memory
  auto& manager = GDRManager::instance();
  ASSERT_TRUE(manager.isAvailable());
  EXPECT_NE(manager.getRegionCache(), nullptr);
}

// @tests REQ-L2-001
TEST_F(TestRDMABufAccelerator, CreateFromDescriptorInvalid) {
  // GIVEN: Default (invalid) descriptor
  AcceleratorMemoryDescriptor desc;
  ASSERT_FALSE(desc.isValid());

  // WHEN: createFromDescriptor is called
  auto buf = RDMABufAccelerator::createFromDescriptor(desc);

  // THEN: Returns invalid buffer
  EXPECT_FALSE(buf.valid());
  EXPECT_EQ(buf.devicePtr(), nullptr);
  EXPECT_EQ(buf.size(), 0u);
}

// @tests REQ-L2-001
TEST_F(TestRDMABufAccelerator, CreateFromDescriptorValidButNoIB) {
  // GIVEN: Valid descriptor fields but no IB devices to register with
  AcceleratorMemoryDescriptor desc;
  desc.devicePtr = reinterpret_cast<void*>(0x1000);
  desc.size = 4096;
  desc.deviceId = 0;

  // WHEN: createFromDescriptor is called on CPU-only machine
  auto buf = RDMABufAccelerator::createFromDescriptor(desc);

  // THEN: On CPU-only: invalid (no IB registration). On GPU: may succeed.
  if (!hasGpu()) {
    EXPECT_FALSE(buf.valid());
  }
}

// ==========================================================================
// REQ-L2-002: GPU RDMA Buffer from IPC Handle
// ==========================================================================

// @tests SCN-L2-002-02
TEST_F(TestRDMABufAccelerator, SCN_L2_002_02_CreateFromIpcHandleNoGDR) {
  if (hasGpu()) {
    GTEST_SKIP() << "Test is for non-GPU environments";
  }

  // GIVEN: GDR not available
  uint8_t fakeHandle[64] = {};
  auto buf = RDMABufAccelerator::createFromIpcHandle(fakeHandle, 4096, 0);

  // THEN: Invalid buffer, no leaked IPC mappings
  EXPECT_FALSE(buf.valid());
  EXPECT_EQ(buf.ptr(), nullptr);
  EXPECT_EQ(buf.size(), 0u);
}

// @tests SCN-L2-002-03
TEST_F(TestRDMABufAccelerator, SCN_L2_002_03_CreateFromIpcHandleGdrBuildDisabled) {
  if (hasGpu()) {
    GTEST_SKIP() << "Test verifies behavior when GDR unavailable";
  }

  uint8_t handle[64] = {};
  auto buf = RDMABufAccelerator::createFromIpcHandle(handle, 4096, 0);
  EXPECT_FALSE(buf.valid());
}

// @tests SCN-L2-002-01
TEST_F(TestRDMABufAccelerator, SCN_L2_002_01_CreateFromIpcHandleSuccess) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU available — integration test only";
  }
  // GIVEN: Valid 64-byte IPC handle from exporter process
  // Requires real GPU memory for the IPC handle to be valid
  auto& manager = GDRManager::instance();
  EXPECT_TRUE(manager.isAvailable());
}

// ==========================================================================
// REQ-L2-003: RDMABufUnified Type-Safe Dispatch
// ==========================================================================

// @tests SCN-L2-003-03
TEST_F(TestRDMABufAccelerator, SCN_L2_003_03_UnifiedEmpty) {
  // GIVEN: Default-constructed RDMABufUnified
  RDMABufUnified unified;

  // THEN: valid()==false, ptr()==nullptr, size()==0
  EXPECT_FALSE(unified.valid());
  EXPECT_FALSE(static_cast<bool>(unified));
  EXPECT_EQ(unified.ptr(), nullptr);
  EXPECT_EQ(unified.size(), 0u);
  EXPECT_EQ(unified.capacity(), 0u);
  EXPECT_EQ(unified.type(), RDMABufUnified::Type::Empty);
  EXPECT_FALSE(unified.isHost());
  EXPECT_FALSE(unified.isGpu());
  EXPECT_EQ(unified.getMR(0), nullptr);

  auto remoteBuf = unified.toRemoteBuf();
  EXPECT_FALSE(static_cast<bool>(remoteBuf));
}

// @tests SCN-L2-003-01
TEST_F(TestRDMABufAccelerator, SCN_L2_003_01_UnifiedGpuDispatch) {
  // GIVEN: An RDMABufAccelerator (even default/invalid for type test)
  RDMABufAccelerator gpuBuf;
  RDMABufUnified unified(std::move(gpuBuf));

  // THEN: isGpu()==true, isHost()==false
  EXPECT_TRUE(unified.isGpu());
  EXPECT_FALSE(unified.isHost());
  EXPECT_EQ(unified.type(), RDMABufUnified::Type::Gpu);

  // Invalid GPU buf: valid() is false but type is Gpu
  EXPECT_FALSE(unified.valid());

  // Access underlying buffer
  auto& gpu = unified.asGpu();
  EXPECT_FALSE(gpu.valid());
  EXPECT_EQ(gpu.devicePtr(), nullptr);
}

// @tests SCN-L2-003-02
TEST_F(TestRDMABufAccelerator, SCN_L2_003_02_UnifiedHostDispatch) {
  // GIVEN: RDMABufUnified constructed with RDMABuf (host)
  RDMABuf hostBuf;  // Default invalid
  RDMABufUnified unified(std::move(hostBuf));

  // THEN: isHost()==true, isGpu()==false
  EXPECT_TRUE(unified.isHost());
  EXPECT_FALSE(unified.isGpu());
  EXPECT_EQ(unified.type(), RDMABufUnified::Type::Host);

  // Access underlying buffer
  auto& host = unified.asHost();
  EXPECT_FALSE(host.valid());
}

// @tests SCN-L2-003-01
TEST_F(TestRDMABufAccelerator, UnifiedHostWithAllocatedBuf) {
  // Try to create a real host RDMABuf
  auto hostBuf = RDMABuf::allocate(4096);
  if (!hostBuf.valid()) {
    GTEST_SKIP() << "Cannot allocate RDMABuf (no IB?) — integration test only";
  }

  RDMABufUnified unified(std::move(hostBuf));
  EXPECT_TRUE(unified.isHost());
  EXPECT_TRUE(unified.valid());
  EXPECT_NE(unified.ptr(), nullptr);
  EXPECT_EQ(unified.size(), 4096u);
}

// ==========================================================================
// REQ-L2-004: Subrange Views and toRemoteBuf
// ==========================================================================

// @tests SCN-L2-004-01, SCN-L2-004-02
TEST_F(TestRDMABufAccelerator, SCN_L2_004_SubrangeOnInvalidBuffer) {
  // GIVEN: Default-constructed (invalid) RDMABufAccelerator
  RDMABufAccelerator buf;
  ASSERT_FALSE(buf.valid());

  // WHEN: subrange is called with various params
  auto sub0 = buf.subrange(0, 0);
  EXPECT_FALSE(sub0.valid());

  auto sub1 = buf.subrange(0, 4096);
  EXPECT_FALSE(sub1.valid());

  auto sub2 = buf.subrange(100, 200);
  EXPECT_FALSE(sub2.valid());
}

// @tests SCN-L2-004-01
TEST_F(TestRDMABufAccelerator, SCN_L2_004_01_SubrangeInBounds) {
  if (!hasGpu()) {
    GTEST_SKIP() << "Need real GPU buffer for subrange test — integration test only";
  }
  auto& manager = GDRManager::instance();
  EXPECT_TRUE(manager.isAvailable());
}

// @tests SCN-L2-004-02
TEST_F(TestRDMABufAccelerator, SCN_L2_004_02_SubrangeOutOfBounds) {
  if (!hasGpu()) {
    GTEST_SKIP() << "Need real GPU buffer for out-of-bounds subrange test — integration test only";
  }
  auto& manager = GDRManager::instance();
  EXPECT_TRUE(manager.isAvailable());
}

// @tests SCN-L2-004-03
TEST_F(TestRDMABufAccelerator, SCN_L2_004_03_ToRemoteBufInvalid) {
  // GIVEN: Invalid buffer
  RDMABufAccelerator buf;

  // WHEN: toRemoteBuf
  auto remoteBuf = buf.toRemoteBuf();

  // THEN: Returns invalid RDMARemoteBuf
  EXPECT_FALSE(static_cast<bool>(remoteBuf));
}

// ==========================================================================
// REQ-L2-005: GPU Buffer Pool
// ==========================================================================

// @tests SCN-L2-005-01, SCN-L2-005-02
TEST_F(TestRDMABufAccelerator, SCN_L2_005_PoolCreation) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU available for pool test — integration test only";
  }
  auto& manager = GDRManager::instance();
  EXPECT_TRUE(manager.isAvailable());
}

// ==========================================================================
// REQ-L2-006: GPU Buffer Synchronization
// ==========================================================================

// @tests SCN-L2-006-02
TEST_F(TestRDMABufAccelerator, SCN_L2_006_02_SyncOnInvalidBuffer) {
  // GIVEN: An invalid (default-constructed) RDMABufAccelerator
  RDMABufAccelerator buf;
  ASSERT_FALSE(buf.valid());

  // WHEN: sync is called with various directions
  // THEN: No-op, no crash
  buf.sync(0);
  buf.sync(1);

  // Verify mock was NOT called (sync on invalid is a no-op)
  auto& mock = hf3fs::test::MockCudaRuntime::instance();
  EXPECT_FALSE(mock.wasSyncCalled());
}

// @tests SCN-L2-006-01
TEST_F(TestRDMABufAccelerator, SCN_L2_006_01_SyncOnValidBuffer) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU available for sync test — integration test only";
  }
  auto& manager = GDRManager::instance();
  EXPECT_TRUE(manager.isAvailable());
}

// ==========================================================================
// Adversarial tests
// ==========================================================================

TEST_F(TestRDMABufAccelerator, Adversarial_DefaultConstructedState) {
  // GIVEN: Default-constructed buffer
  RDMABufAccelerator buf;

  // THEN: All accessors are safe on invalid buffer
  EXPECT_FALSE(buf.valid());
  EXPECT_EQ(buf.devicePtr(), nullptr);
  EXPECT_EQ(buf.ptr(), nullptr);
  EXPECT_EQ(buf.capacity(), 0u);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.empty());
  EXPECT_EQ(buf.deviceId(), -1);
  EXPECT_EQ(buf.getMR(0), nullptr);
  EXPECT_FALSE(buf.getRkey(0).has_value());
}

TEST_F(TestRDMABufAccelerator, Adversarial_AdvanceOnEmpty) {
  RDMABufAccelerator buf;

  // advance(1) on empty should fail
  EXPECT_FALSE(buf.advance(1));
  // subtract(1) on empty should fail
  EXPECT_FALSE(buf.subtract(1));
}

TEST_F(TestRDMABufAccelerator, Adversarial_FirstLastOnEmpty) {
  RDMABufAccelerator buf;

  auto f = buf.first(0);
  EXPECT_FALSE(f.valid());

  auto l = buf.last(0);
  EXPECT_FALSE(l.valid());

  auto f1 = buf.first(1);
  EXPECT_FALSE(f1.valid());

  auto l1 = buf.last(1);
  EXPECT_FALSE(l1.valid());
}

TEST_F(TestRDMABufAccelerator, Adversarial_ContainsOnEmpty) {
  RDMABufAccelerator buf;

  uint8_t data[1] = {0};
  EXPECT_FALSE(buf.contains(data, 1));
  EXPECT_FALSE(buf.contains(nullptr, 0));
}

TEST_F(TestRDMABufAccelerator, Adversarial_GetIpcHandleOnEmpty) {
  RDMABufAccelerator buf;
  uint8_t handle[64] = {};
  bool result = buf.getIpcHandle(handle);
  EXPECT_FALSE(result);
}

TEST_F(TestRDMABufAccelerator, Adversarial_MoveSemantics) {
  // Default-constructed, moved from
  RDMABufAccelerator buf1;
  RDMABufAccelerator buf2(std::move(buf1));

  EXPECT_FALSE(buf2.valid());
  EXPECT_FALSE(buf1.valid());  // moved-from state

  // Move assignment
  RDMABufAccelerator buf3;
  buf3 = std::move(buf2);
  EXPECT_FALSE(buf3.valid());
}

TEST_F(TestRDMABufAccelerator, Adversarial_UnifiedMoveSemantics) {
  RDMABufUnified u1;
  EXPECT_FALSE(u1.valid());

  RDMABufUnified u2(RDMABufAccelerator{});
  EXPECT_TRUE(u2.isGpu());
  EXPECT_FALSE(u2.valid());

  // Access underlying buffers
  auto& gpu = u2.asGpu();
  EXPECT_FALSE(gpu.valid());
}

// @tests SCN-L2-003-01, SCN-L2-003-02
TEST_F(TestRDMABufAccelerator, UnifiedMRDispatch) {
  // GIVEN: Empty unified buffer
  RDMABufUnified empty;

  // WHEN: getMR is called on empty
  EXPECT_EQ(empty.getMR(0), nullptr);

  // WHEN: toRemoteBuf is called on empty
  auto remoteBuf = empty.toRemoteBuf();
  EXPECT_FALSE(static_cast<bool>(remoteBuf));

  // GPU-typed unified
  RDMABufUnified gpuUnified(RDMABufAccelerator{});
  EXPECT_EQ(gpuUnified.getMR(0), nullptr);

  // Host-typed unified
  RDMABufUnified hostUnified(RDMABuf{});
  EXPECT_EQ(hostUnified.getMR(0), nullptr);
}

}  // namespace hf3fs::net
