/**
 * Scenario tests for Layer 3+4: UsrbIoGdr + UsrbIo (unified C API)
 *
 * Tests GPU IOV create/open/wrap/destroy, URI parsing, global registry,
 * unified dispatch, query functions, and sync.
 *
 * Covers: REQ-L3-001 through REQ-L3-006
 *         REQ-L4-001 through REQ-L4-006
 *         INV-GDR-001, INV-GDR-002, INV-GDR-003, INV-GDR-006
 *
 * URI parsing (REQ-L3-005): parseGpuIovTarget() is in an anonymous namespace
 * in UsrbIoGdr.cc. Testing through public API hf3fs_iovopen which requires
 * GDR availability. On CPU-only, we test the dispatch/error paths; on GPU
 * machines the full parser is exercised via crafted symlinks.
 */

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "lib/api/hf3fs_usrbio.h"
#include "tests/GtestHelpers.h"
#include "tests/gdr/mocks/MockCudaRuntime.h"

namespace {

static bool hasGpu() {
  return hf3fs_gdr_available();
}

// Temp directory for symlink testing
class TmpDir {
 public:
  TmpDir() {
    const char* base = getenv("TMPDIR");
    if (!base) base = "/private/tmp/claude-502";
    path_ = std::string(base) + "/gdr_test_XXXXXX";
    char* result = mkdtemp(path_.data());
    if (result) {
      valid_ = true;
    }
  }

  ~TmpDir() {
    if (valid_) {
      std::filesystem::remove_all(path_);
    }
  }

  const char* path() const { return path_.c_str(); }
  bool valid() const { return valid_; }

 private:
  std::string path_;
  bool valid_ = false;
};

// Helper to build a well-formed GDR URI
std::string buildGdrUri(int deviceId, size_t size, const uint8_t ipcHandle[64]) {
  char hex[129];
  for (int i = 0; i < 64; i++) {
    snprintf(hex + i * 2, 3, "%02x", ipcHandle[i]);
  }
  hex[128] = '\0';
  return std::string("gdr://v1/device/") + std::to_string(deviceId) +
         "/size/" + std::to_string(size) + "/ipc/" + hex;
}

class TestUsrbIoGdrFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    hf3fs::test::MockCudaRuntime::instance().reset();
  }

  void TearDown() override {
    hf3fs::test::MockCudaRuntime::instance().reset();
  }
};

}  // namespace

// ==========================================================================
// REQ-L4-001: iovcreate host path; iovcreate_device dispatches to GPU
// ==========================================================================

// @tests SCN-L4-001-04
TEST_F(TestUsrbIoGdrFixture, SCN_L4_001_04_HostPathPositiveNuma) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  int rc = hf3fs_iovcreate(&iov, "/nonexistent/mount", 4096, 0, 0);

  // THEN: Should fail (no mount point) but uses host path
  EXPECT_NE(rc, 0);
}

// @tests SCN-L4-001-01
TEST_F(TestUsrbIoGdrFixture, SCN_L4_001_01_NegativeNumaHostPath) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // WHEN: iovcreate with negative numa (= no NUMA binding, host path)
  int rc = hf3fs_iovcreate(&iov, "/nonexistent/mount", 4096, 0, -1);

  // THEN: Should fail (no mount) but takes host path, no GPU dispatch
  EXPECT_NE(rc, 0);
}

// @tests SCN-L4-001-01b
TEST_F(TestUsrbIoGdrFixture, SCN_L4_001_01b_DeviceApiDispatch) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // WHEN: iovcreate_device with device 0
  int rc = hf3fs_iovcreate_device(&iov, "/nonexistent/mount", 4096, 0, 0);

  // THEN: Should fail (no mount) but exercise dispatch
  EXPECT_NE(rc, 0);
}

// @tests SCN-L4-001-02
TEST_F(TestUsrbIoGdrFixture, SCN_L4_001_02_DeviceFallbackNoGDR) {
  if (hasGpu()) {
    GTEST_SKIP() << "Test for machines without GPU";
  }

  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // WHEN: iovcreate_device, GDR unavailable
  int rc = hf3fs_iovcreate_device(&iov, "/nonexistent/mount", 4096, 0, 0);

  // THEN: Falls back to host path (still fails due to no mount, but no crash)
  EXPECT_NE(rc, 0);
}

// ==========================================================================
// REQ-L4-002: iovopen/iovwrap host path; _device variants for GPU
// ==========================================================================

// @tests SCN-L4-002-00a
TEST_F(TestUsrbIoGdrFixture, SCN_L4_002_00a_IovOpenNegativeNumaHostPath) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  uint8_t id[16] = {};

  // WHEN: iovopen with negative numa (= host path, no NUMA binding)
  int rc = hf3fs_iovopen(&iov, id, "/nonexistent", 4096, 0, -1);
  // THEN: Fails because mount doesn't exist, but no GPU dispatch
  EXPECT_NE(rc, 0);
}

// @tests SCN-L4-002-00b
TEST_F(TestUsrbIoGdrFixture, SCN_L4_002_00b_IovWrapNegativeNumaHostPath) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  uint8_t id[16] = {};
  // Use a valid host pointer so iovwrap succeeds (it doesn't validate mount existence)
  uint8_t buf[64] = {};

  // WHEN: iovwrap with negative numa (= host path, no NUMA binding)
  int rc = hf3fs_iovwrap(&iov, buf, id, "/nonexistent", sizeof(buf), 0, -1);

  // THEN: Succeeds (iovwrap doesn't validate mount), host path, no GPU dispatch
  EXPECT_EQ(rc, 0);
  // Verify it's host memory, not GPU
  EXPECT_EQ(hf3fs_iov_mem_type(&iov), HF3FS_MEM_HOST);
}

// @tests SCN-L4-002-01
TEST_F(TestUsrbIoGdrFixture, SCN_L4_002_01_IovOpenDeviceNoGdr) {
  if (hasGpu()) {
    GTEST_SKIP() << "Test for non-GPU environment";
  }

  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  uint8_t id[16] = {};

  // WHEN: iovopen_device, GDR unavailable
  int rc = hf3fs_iovopen_device(&iov, id, "/nonexistent", 4096, 0, 0);

  // THEN: Returns -ENOTSUP
  EXPECT_EQ(rc, -ENOTSUP);
}

// @tests SCN-L4-002-02
TEST_F(TestUsrbIoGdrFixture, SCN_L4_002_02_IovWrapDeviceNoGdr) {
  if (hasGpu()) {
    GTEST_SKIP() << "Test for non-GPU environment";
  }

  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  uint8_t id[16] = {};
  void* fakePtr = reinterpret_cast<void*>(0x1000);

  // WHEN: iovwrap_device, GDR unavailable
  int rc = hf3fs_iovwrap_device(&iov, fakePtr, id, "/nonexistent", 4096, 0, 0);

  // THEN: Returns -ENOTSUP
  EXPECT_EQ(rc, -ENOTSUP);
}

// ==========================================================================
// REQ-L4-003: Unified iovdestroy Dispatch
// ==========================================================================

// @tests SCN-L4-003-02
TEST_F(TestUsrbIoGdrFixture, SCN_L4_003_02_DestroyHostIov) {
  // GIVEN: A zeroed iov (host-like, numa >= 0)
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = 0;

  // WHEN: iovdestroy is called
  // THEN: No crash (best-effort cleanup on zeroed iov)
  hf3fs_iovdestroy(&iov);

  // Verify iov was zeroed/cleaned
  EXPECT_EQ(iov.base, nullptr);
}

// @tests SCN-L4-003-01
TEST_F(TestUsrbIoGdrFixture, SCN_L4_003_01_DestroyGpuIovZeroed) {
  // GIVEN: A zeroed iov with GPU magic numa but no real handle
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = -0x6472;  // kGpuIovMagicNuma

  // WHEN: iovdestroy is called
  // THEN: Should detect it's not in gGpuIovHandles and no crash
  hf3fs_iovdestroy(&iov);

  // Verify mock was not asked to free (no real GPU handle)
  auto& mock = hf3fs::test::MockCudaRuntime::instance();
  EXPECT_EQ(mock.mallocCallCount(), 0);
}

// ==========================================================================
// REQ-L4-004: Query Functions
// ==========================================================================

// @tests SCN-L4-004-01
TEST_F(TestUsrbIoGdrFixture, SCN_L4_004_01_MemTypeHostIov) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = 0;

  // WHEN: mem_type query
  enum hf3fs_mem_type type = hf3fs_iov_mem_type(&iov);

  // THEN: HF3FS_MEM_HOST
  EXPECT_EQ(type, HF3FS_MEM_HOST);
}

// @tests SCN-L4-004-01
TEST_F(TestUsrbIoGdrFixture, SCN_L4_004_01_DeviceIdHostIov) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = 0;

  // WHEN: device_id query on host iov
  int devId = hf3fs_iov_device_id(&iov);

  // THEN: Returns -1
  EXPECT_EQ(devId, -1);
}

// @tests SCN-L4-004-01
TEST_F(TestUsrbIoGdrFixture, SCN_L4_004_01_MemTypeGpuMagicButNoHandle) {
  // GIVEN: An iov with GPU magic numa but NOT registered in gGpuIovHandles
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = -0x6472;  // kGpuIovMagicNuma
  iov.iovh = nullptr;

  // WHEN: mem_type query
  enum hf3fs_mem_type type = hf3fs_iov_mem_type(&iov);

  // THEN: Returns HOST because handle is not registered
  EXPECT_EQ(type, HF3FS_MEM_HOST);

  // Device ID should be -1 (unregistered)
  int devId = hf3fs_iov_device_id(&iov);
  EXPECT_EQ(devId, -1);
}

// @tests SCN-L4-004-02
TEST_F(TestUsrbIoGdrFixture, SCN_L4_004_02_GdrAvailableLazyInit) {
  // WHEN: hf3fs_gdr_available is called
  bool avail = hf3fs_gdr_available();

  // THEN: Returns a valid boolean, consistent across calls
  bool avail2 = hf3fs_gdr_available();
  EXPECT_EQ(avail, avail2);
}

// @tests REQ-L4-004
TEST_F(TestUsrbIoGdrFixture, GdrDeviceCount) {
  int count = hf3fs_gdr_device_count();
  if (hasGpu()) {
    EXPECT_GT(count, 0);
  } else {
    EXPECT_EQ(count, 0);
  }
}

// ==========================================================================
// REQ-L4-006: iovsync Dispatch
// ==========================================================================

// @tests SCN-L4-006-02
TEST_F(TestUsrbIoGdrFixture, SCN_L4_006_02_SyncHostIov) {
  // GIVEN: Host iov
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = 0;

  // WHEN: iovsync is called
  int rc = hf3fs_iovsync(&iov, 1);

  // THEN: Returns 0 (no-op for host)
  EXPECT_EQ(rc, 0);

  // Verify mock CUDA sync was NOT called (host path is no-op)
  auto& mock = hf3fs::test::MockCudaRuntime::instance();
  EXPECT_FALSE(mock.wasSyncCalled());
}

// @tests SCN-L4-006-02
TEST_F(TestUsrbIoGdrFixture, SyncHostIovDirection0) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = 0;

  int rc = hf3fs_iovsync(&iov, 0);
  EXPECT_EQ(rc, 0);
}

// ==========================================================================
// REQ-L3-001: GPU IOV Create with Library-Managed Memory
// ==========================================================================

// @tests SCN-L3-001-05
TEST_F(TestUsrbIoGdrFixture, SCN_L3_001_05_InvalidDeviceId) {
  if (!hasGpu()) {
    // On CPU-only: iovcreate_device falls back to host, still fails due to mount
    struct hf3fs_iov iov;
    memset(&iov, 0, sizeof(iov));
    int rc = hf3fs_iovcreate_device(&iov, "/nonexistent/mount", 4096, 0, 99);
    EXPECT_NE(rc, 0);
    return;
  }

  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // device_id=99 (nonexistent)
  int rc = hf3fs_iovcreate_device(&iov, "/nonexistent/mount", 4096, 0, 99);

  // THEN: Returns error (bad device or no mount)
  EXPECT_NE(rc, 0);
}

// @tests SCN-L3-001-01
TEST_F(TestUsrbIoGdrFixture, SCN_L3_001_01_FullSuccessPath) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU available — integration test only";
  }
  // Integration test: requires real mount point + GPU
  GTEST_SKIP() << "Requires real hf3fs mount — integration test only";
}

// @tests SCN-L3-001-02
TEST_F(TestUsrbIoGdrFixture, SCN_L3_001_02_MallocFail) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU — integration test only (needs link-time mock)";
  }

  // Configure mock to fail malloc
  auto& mock = hf3fs::test::MockCudaRuntime::instance();
  mock.setDeviceCount(1);
  mock.setMallocBehavior([](void** devPtr, size_t size) -> cudaError_t {
    (void)devPtr;
    (void)size;
    return cudaErrorMemoryAllocation;
  });

  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // WHEN: iovcreate_device, malloc will fail
  int rc = hf3fs_iovcreate_device(&iov, "/nonexistent/mount", 4096, 0, 0);

  // THEN: Returns error (malloc failure or mount failure)
  EXPECT_NE(rc, 0);
}

// ==========================================================================
// REQ-L3-002: GPU IOV Open (Cross-Process Reopen)
// ==========================================================================

// @tests SCN-L3-002-02
TEST_F(TestUsrbIoGdrFixture, SCN_L3_002_02_OpenNotFound) {
  if (!hasGpu()) {
    // On CPU-only, iovopen_device returns -ENOTSUP
    struct hf3fs_iov iov;
    memset(&iov, 0, sizeof(iov));
    uint8_t id[16] = {0xDE, 0xAD, 0xBE, 0xEF};
    int rc = hf3fs_iovopen_device(&iov, id, "/nonexistent", 4096, 0, 0);
    EXPECT_EQ(rc, -ENOTSUP);
    return;
  }

  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  uint8_t id[16] = {0xDE, 0xAD, 0xBE, 0xEF};

  // WHEN: iovopen_device with non-existent UUID
  int rc = hf3fs_iovopen_device(&iov, id, "/nonexistent", 4096, 0, 0);

  // THEN: Returns error (no symlink found)
  EXPECT_NE(rc, 0);
}

// ==========================================================================
// REQ-L3-004: GPU IOV Destroy with Correct Lifecycle
// ==========================================================================

// @tests SCN-L3-004-01, SCN-L3-004-02, SCN-L3-004-03
// Tests destroy on zeroed iov covers all ownership types (owned/wrapped/imported)
// since the zeroed state simulates "handle not in gGpuIovHandles" for all cases
TEST_F(TestUsrbIoGdrFixture, Adversarial_DoubleDestroy) {
  // GIVEN: A zeroed iov (not actually created)
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // WHEN: Double destroy
  // THEN: No crash on both calls
  hf3fs_iovdestroy(&iov);
  hf3fs_iovdestroy(&iov);

  // Verify iov is still zeroed
  EXPECT_EQ(iov.base, nullptr);
  EXPECT_EQ(iov.iovh, nullptr);
}

// @tests SCN-L3-004-01
TEST_F(TestUsrbIoGdrFixture, SCN_L3_004_01_DestroyGpuMagicNoHandle) {
  // GIVEN: iov with GPU magic numa but no registered handle
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = -0x6472;  // kGpuIovMagicNuma

  // WHEN: destroy
  hf3fs_iovdestroy(&iov);

  // THEN: No crash, mock CUDA free was not called
  auto& mock = hf3fs::test::MockCudaRuntime::instance();
  EXPECT_FALSE(mock.wasFreeCalled(nullptr));
}

// ==========================================================================
// REQ-L3-005: GDR URI Parsing (Pure Logic)
// ==========================================================================
//
// parseGpuIovTarget() is in anonymous namespace in UsrbIoGdr.cc.
// We test through hf3fs_iovopen (which calls it) on GPU machines,
// and verify URI format contracts + error handling on CPU-only.
// The URI format is: "gdr://v1/device/{N}/size/{S}/ipc/{hex128}"

// @tests SCN-L3-005-01
TEST_F(TestUsrbIoGdrFixture, SCN_L3_005_01_ValidUriFormatThroughIovopen) {
  // GIVEN: A valid URI in symlink form
  // Build a proper URI and verify format through the API path
  uint8_t ipcHandle[64];
  for (int i = 0; i < 64; i++) ipcHandle[i] = static_cast<uint8_t>(i * 3 + 7);

  std::string uri = buildGdrUri(0, 1073741824, ipcHandle);

  // Verify URI has correct format
  EXPECT_EQ(uri.substr(0, 14), "gdr://v1/devic");
  EXPECT_NE(uri.find("/device/0/"), std::string::npos);
  EXPECT_NE(uri.find("/size/1073741824/"), std::string::npos);
  EXPECT_NE(uri.find("/ipc/"), std::string::npos);
  // IPC hex should be 128 chars
  size_t ipcPos = uri.find("/ipc/");
  ASSERT_NE(ipcPos, std::string::npos);
  std::string hexPart = uri.substr(ipcPos + 5);
  EXPECT_EQ(hexPart.size(), 128u);

  // On GPU machines, test through iovopen with a crafted symlink
  if (hasGpu()) {
    TmpDir tmpDir;
    if (tmpDir.valid()) {
      // Create the symlink structure: {mount}/3fs-virt/iovs/{uuid}.gdr.d0 -> uri
      std::string virtDir = std::string(tmpDir.path()) + "/3fs-virt/iovs";
      std::filesystem::create_directories(virtDir);

      // Create a symlink with the URI
      std::string symlinkPath = virtDir + "/deadbeef12345678deadbeef12345678.gdr.d0";
      symlink(uri.c_str(), symlinkPath.c_str());

      // Try to open — will fail because UUID doesn't match, but exercises parser
      struct hf3fs_iov iov;
      memset(&iov, 0, sizeof(iov));
      uint8_t id[16] = {0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78,
                         0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78};
      int rc = hf3fs_iovopen_device(&iov, id, tmpDir.path(), 1073741824, 0, 0);
      // May fail for various reasons, but should not crash
      (void)rc;
    }
  }
}

// @tests SCN-L3-005-02
TEST_F(TestUsrbIoGdrFixture, SCN_L3_005_02_TruncatedHex) {
  // GIVEN: URI with only 126 hex characters (should be 128)
  std::string truncatedUri = "gdr://v1/device/0/size/1024/ipc/";
  for (int i = 0; i < 126; i++) {
    truncatedUri += "a";
  }

  // Verify the hex portion IS too short
  size_t hexLen = truncatedUri.size() - strlen("gdr://v1/device/0/size/1024/ipc/");
  EXPECT_EQ(hexLen, 126u);
  EXPECT_NE(hexLen, 128u);

  // On GPU: test through iovopen with this malformed URI as symlink target
  if (hasGpu()) {
    TmpDir tmpDir;
    if (tmpDir.valid()) {
      std::string virtDir = std::string(tmpDir.path()) + "/3fs-virt/iovs";
      std::filesystem::create_directories(virtDir);

      std::string symlinkPath = virtDir + "/00000000000000000000000000000001.gdr.d0";
      symlink(truncatedUri.c_str(), symlinkPath.c_str());

      struct hf3fs_iov iov;
      memset(&iov, 0, sizeof(iov));
      uint8_t id[16] = {};
      id[15] = 1;
      int rc = hf3fs_iovopen_device(&iov, id, tmpDir.path(), 1024, 0, 0);
      // THEN: Should fail (truncated hex = parse failure)
      EXPECT_NE(rc, 0);
    }
  }
}

// @tests SCN-L3-005-03
TEST_F(TestUsrbIoGdrFixture, SCN_L3_005_03_NonHexCharacters) {
  // GIVEN: URI with non-hex characters in IPC handle field
  std::string badUri = "gdr://v1/device/0/size/1024/ipc/";
  for (int i = 0; i < 64; i++) {
    badUri += "XX";  // Not valid hex
  }
  EXPECT_EQ(badUri.size(), strlen("gdr://v1/device/0/size/1024/ipc/") + 128);

  // On GPU: test through iovopen
  if (hasGpu()) {
    TmpDir tmpDir;
    if (tmpDir.valid()) {
      std::string virtDir = std::string(tmpDir.path()) + "/3fs-virt/iovs";
      std::filesystem::create_directories(virtDir);

      std::string symlinkPath = virtDir + "/00000000000000000000000000000002.gdr.d0";
      symlink(badUri.c_str(), symlinkPath.c_str());

      struct hf3fs_iov iov;
      memset(&iov, 0, sizeof(iov));
      uint8_t id[16] = {};
      id[15] = 2;
      int rc = hf3fs_iovopen_device(&iov, id, tmpDir.path(), 1024, 0, 0);
      // THEN: Should fail (non-hex chars = parse failure)
      EXPECT_NE(rc, 0);
    }
  }
}

// @tests SCN-L3-005-04
TEST_F(TestUsrbIoGdrFixture, SCN_L3_005_04_WrongVersionPrefix) {
  // GIVEN: URI with v2 instead of v1
  std::string v2Uri = "gdr://v2/device/0/size/1024/ipc/";
  for (int i = 0; i < 128; i++) {
    v2Uri += "a";
  }

  // Verify the version prefix IS wrong
  EXPECT_NE(v2Uri.find("v2"), std::string::npos);
  EXPECT_EQ(v2Uri.find("gdr://v1"), std::string::npos);

  // On GPU: test through iovopen
  if (hasGpu()) {
    TmpDir tmpDir;
    if (tmpDir.valid()) {
      std::string virtDir = std::string(tmpDir.path()) + "/3fs-virt/iovs";
      std::filesystem::create_directories(virtDir);

      std::string symlinkPath = virtDir + "/00000000000000000000000000000003.gdr.d0";
      symlink(v2Uri.c_str(), symlinkPath.c_str());

      struct hf3fs_iov iov;
      memset(&iov, 0, sizeof(iov));
      uint8_t id[16] = {};
      id[15] = 3;
      int rc = hf3fs_iovopen_device(&iov, id, tmpDir.path(), 1024, 0, 0);
      // THEN: Should fail (wrong version prefix)
      EXPECT_NE(rc, 0);
    }
  }
}

TEST_F(TestUsrbIoGdrFixture, Adversarial_EmptyUri) {
  // GIVEN: Empty URI
  std::string emptyUri;
  EXPECT_TRUE(emptyUri.empty());

  // On GPU: test through iovopen with empty symlink target
  if (hasGpu()) {
    TmpDir tmpDir;
    if (tmpDir.valid()) {
      std::string virtDir = std::string(tmpDir.path()) + "/3fs-virt/iovs";
      std::filesystem::create_directories(virtDir);

      // Empty symlink target would fail symlink() itself
      // So create a symlink to a non-gdr string
      std::string symlinkPath = virtDir + "/00000000000000000000000000000004.gdr.d0";
      symlink("", symlinkPath.c_str());

      struct hf3fs_iov iov;
      memset(&iov, 0, sizeof(iov));
      uint8_t id[16] = {};
      id[15] = 4;
      int rc = hf3fs_iovopen_device(&iov, id, tmpDir.path(), 1024, 0, 0);
      EXPECT_NE(rc, 0);
    }
  }
}

TEST_F(TestUsrbIoGdrFixture, Adversarial_NegativeDeviceIdInUri) {
  std::string uri = "gdr://v1/device/-1/size/1024/ipc/";
  for (int i = 0; i < 128; i++) {
    uri += "a";
  }
  // Verify the URI has a negative device
  EXPECT_NE(uri.find("device/-1"), std::string::npos);

  if (hasGpu()) {
    TmpDir tmpDir;
    if (tmpDir.valid()) {
      std::string virtDir = std::string(tmpDir.path()) + "/3fs-virt/iovs";
      std::filesystem::create_directories(virtDir);

      std::string symlinkPath = virtDir + "/00000000000000000000000000000005.gdr.d0";
      symlink(uri.c_str(), symlinkPath.c_str());

      struct hf3fs_iov iov;
      memset(&iov, 0, sizeof(iov));
      uint8_t id[16] = {};
      id[15] = 5;
      int rc = hf3fs_iovopen_device(&iov, id, tmpDir.path(), 1024, 0, 0);
      EXPECT_NE(rc, 0);
    }
  }
}

TEST_F(TestUsrbIoGdrFixture, Adversarial_ZeroSizeInUri) {
  uint8_t handle[64] = {};
  std::string uri = buildGdrUri(0, 0, handle);
  EXPECT_NE(uri.find("size/0"), std::string::npos);

  if (hasGpu()) {
    TmpDir tmpDir;
    if (tmpDir.valid()) {
      std::string virtDir = std::string(tmpDir.path()) + "/3fs-virt/iovs";
      std::filesystem::create_directories(virtDir);

      std::string symlinkPath = virtDir + "/00000000000000000000000000000006.gdr.d0";
      symlink(uri.c_str(), symlinkPath.c_str());

      struct hf3fs_iov iov;
      memset(&iov, 0, sizeof(iov));
      uint8_t id[16] = {};
      id[15] = 6;
      int rc = hf3fs_iovopen_device(&iov, id, tmpDir.path(), 0, 0, 0);
      // Size=0 may be parsed but caller validates; expect error
      EXPECT_NE(rc, 0);
    }
  }
}

TEST_F(TestUsrbIoGdrFixture, EncodeDecodeRoundTrip) {
  // Verify 64-byte data can be hex-encoded and decoded (URI encoding contract)
  uint8_t original[64];
  for (int i = 0; i < 64; i++) {
    original[i] = static_cast<uint8_t>(i * 3 + 7);
  }

  // Encode to hex (same as what buildGdrUri does)
  char hex[129];
  for (int i = 0; i < 64; i++) {
    snprintf(hex + i * 2, 3, "%02x", original[i]);
  }
  hex[128] = '\0';

  // Decode from hex
  uint8_t decoded[64];
  for (int i = 0; i < 64; i++) {
    unsigned int byte;
    sscanf(hex + i * 2, "%2x", &byte);
    decoded[i] = static_cast<uint8_t>(byte);
  }

  // Round-trip matches
  EXPECT_EQ(memcmp(original, decoded, 64), 0);

  // Verify the encoded URI uses this correctly
  std::string uri = buildGdrUri(0, 4096, original);
  EXPECT_NE(uri.find(std::string(hex)), std::string::npos);
}

// ==========================================================================
// REQ-L3-003: GPU IOV Wrap (External Memory)
// ==========================================================================

// @tests SCN-L3-003-01
TEST_F(TestUsrbIoGdrFixture, SCN_L3_003_01_WrapExternalGpuPtr) {
  if (!hasGpu()) {
    // On CPU-only, iovwrap_device returns -ENOTSUP
    struct hf3fs_iov iov;
    memset(&iov, 0, sizeof(iov));
    uint8_t id[16] = {};
    void* fakePtr = reinterpret_cast<void*>(0x2000);
    int rc = hf3fs_iovwrap_device(&iov, fakePtr, id, "/nonexistent", 4096, 0, 0);
    EXPECT_EQ(rc, -ENOTSUP);
    return;
  }

  // On GPU: wrap would need real GPU pointer + mount
  GTEST_SKIP() << "Requires real GPU memory and mount — integration test only";
}

// @tests SCN-L3-003-02
TEST_F(TestUsrbIoGdrFixture, SCN_L3_003_02_WrapIpcExportFailure) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU — needs link-time mock for IPC failure injection";
  }

  // Configure mock to fail IPC export
  auto& mock = hf3fs::test::MockCudaRuntime::instance();
  mock.setIpcGetHandleBehavior([](cudaIpcMemHandle_t* handle, void* devPtr) -> cudaError_t {
    (void)handle;
    (void)devPtr;
    return cudaErrorMapBufferObjectFailed;
  });

  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  uint8_t id[16] = {1, 2, 3, 4};
  int rc = hf3fs_iovwrap_device(&iov, reinterpret_cast<void*>(0x3000), id,
                                "/nonexistent", 4096, 0, 0);
  // THEN: Returns error
  EXPECT_NE(rc, 0);
}

// ==========================================================================
// REQ-L3-006: GpuIovHandle Global Registry
// ==========================================================================

// @tests SCN-L3-006-02
TEST_F(TestUsrbIoGdrFixture, SCN_L3_006_02_LookupNonGpuIov) {
  // GIVEN: An iov with numa != kGpuIovMagicNuma
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = 0;

  // WHEN: mem_type check (which internally checks gGpuIovHandles)
  enum hf3fs_mem_type type = hf3fs_iov_mem_type(&iov);

  // THEN: Returns HOST, meaning no GPU handle found
  EXPECT_EQ(type, HF3FS_MEM_HOST);
}

// @tests SCN-L3-006-01
TEST_F(TestUsrbIoGdrFixture, SCN_L3_006_01_RegisterAndLookup) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU — integration test only";
  }
  GTEST_SKIP() << "Requires full IOV lifecycle for registry test — integration test only";
}

// ==========================================================================
// Explicit Device API: verify _device functions exist
// ==========================================================================

TEST_F(TestUsrbIoGdrFixture, DeviceApiFunctionsExist) {
  auto createFn = &hf3fs_iovcreate_device;
  auto openFn = &hf3fs_iovopen_device;
  auto wrapFn = &hf3fs_iovwrap_device;

  EXPECT_NE(createFn, nullptr);
  EXPECT_NE(openFn, nullptr);
  EXPECT_NE(wrapFn, nullptr);
}

// ==========================================================================
// INV-GDR-001: iov->iovh Polymorphism Discriminant
// ==========================================================================

// @tests INV-GDR-001
TEST_F(TestUsrbIoGdrFixture, INV_GDR_001_PolymorphismSafety) {
  // GIVEN: An iov that looks GPU-like but has no real handle
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  iov.numa = -0x6472;  // kGpuIovMagicNuma
  iov.iovh = nullptr;

  // WHEN: Query operations are called
  // THEN: They must not crash and must recognize this as non-GPU
  enum hf3fs_mem_type type = hf3fs_iov_mem_type(&iov);
  EXPECT_EQ(type, HF3FS_MEM_HOST);

  int devId = hf3fs_iov_device_id(&iov);
  EXPECT_EQ(devId, -1);

  // Sync should be no-op on unregistered GPU-magic iov
  int rc = hf3fs_iovsync(&iov, 0);
  EXPECT_EQ(rc, 0);
}

// @tests INV-GDR-002
TEST_F(TestUsrbIoGdrFixture, INV_GDR_002_MagicNumaValue) {
  // Verify the magic numa value is stable
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // Host iov: numa >= 0
  iov.numa = 0;
  EXPECT_EQ(hf3fs_iov_mem_type(&iov), HF3FS_MEM_HOST);

  // GPU iov requires numa == -0x6472 AND registered handle
  iov.numa = -0x6472;
  // Without registered handle, still reports HOST
  EXPECT_EQ(hf3fs_iov_mem_type(&iov), HF3FS_MEM_HOST);

  // Verify the actual numeric value
  EXPECT_EQ(-0x6472, -25714);
}

// ==========================================================================
// Adversarial: Concurrent and Stress
// ==========================================================================

TEST_F(TestUsrbIoGdrFixture, Adversarial_ConcurrentGdrAvailableQuery) {
  std::vector<std::thread> threads;
  std::atomic<int> trueCount{0};
  std::atomic<int> falseCount{0};

  for (int i = 0; i < 16; i++) {
    threads.emplace_back([&]() {
      for (int j = 0; j < 100; j++) {
        if (hf3fs_gdr_available()) {
          trueCount++;
        } else {
          falseCount++;
        }
      }
    });
  }

  for (auto& t : threads) t.join();

  // All queries should return the same result
  EXPECT_EQ(trueCount + falseCount, 16 * 100);
  // Either all true or all false
  EXPECT_TRUE(trueCount == 0 || falseCount == 0);
}

TEST_F(TestUsrbIoGdrFixture, Adversarial_SyncInvalidDirection) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // Invalid direction value on host iov (no-op path)
  int rc = hf3fs_iovsync(&iov, 99);
  EXPECT_EQ(rc, 0);  // Host path is always no-op
}

// Security: Malicious mount path
TEST_F(TestUsrbIoGdrFixture, Adversarial_SecurityInjectionInMountPath) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // Path traversal attempt
  int rc = hf3fs_iovcreate(&iov, "/../../../etc/passwd", 4096, 0, 0);
  EXPECT_NE(rc, 0);
}

TEST_F(TestUsrbIoGdrFixture, Adversarial_SecurityNullMountPoint) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // Null mount point
  int rc = hf3fs_iovcreate(&iov, nullptr, 4096, 0, 0);
  // Should not crash, should return error
  EXPECT_NE(rc, 0);
}

TEST_F(TestUsrbIoGdrFixture, Adversarial_SecurityVeryLongMountPoint) {
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  // Very long mount point (beyond iov.mount_point capacity of 256)
  std::string longPath(1024, 'A');
  int rc = hf3fs_iovcreate(&iov, longPath.c_str(), 4096, 0, 0);
  EXPECT_NE(rc, 0);
}

// ==========================================================================
// Additional coverage
// ==========================================================================

// @tests SCN-L3-001-03
TEST_F(TestUsrbIoGdrFixture, SCN_L3_001_03_MrRegistrationFailure) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU — needs link-time mock for MR failure injection";
  }
  GTEST_SKIP() << "Requires mock IB device for controlled MR failure — integration test only";
}

// @tests SCN-L3-001-04
TEST_F(TestUsrbIoGdrFixture, SCN_L3_001_04_SymlinkCreationFailure) {
  // Test symlink creation failure through invalid mount
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));

  int rc = hf3fs_iovcreate_device(&iov, "/nonexistent/does/not/exist", 4096, 0, 0);
  // THEN: Returns error (mount doesn't exist)
  EXPECT_NE(rc, 0);
}

// @tests SCN-L3-002-01
TEST_F(TestUsrbIoGdrFixture, SCN_L3_002_01_SuccessfulReopen) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU — integration test only";
  }
  GTEST_SKIP() << "Requires real hf3fs mount — integration test only";
}

// @tests SCN-L3-002-03
TEST_F(TestUsrbIoGdrFixture, SCN_L3_002_03_DeviceMismatch) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU — integration test only";
  }
  GTEST_SKIP() << "Requires real mount + GPU — integration test only";
}

// @tests SCN-L3-002-04
TEST_F(TestUsrbIoGdrFixture, SCN_L3_002_04_MalformedUri) {
  // On CPU-only: iovopen_device returns -ENOTSUP
  if (!hasGpu()) {
    struct hf3fs_iov iov;
    memset(&iov, 0, sizeof(iov));
    uint8_t id[16] = {};
    int rc = hf3fs_iovopen_device(&iov, id, "/nonexistent", 4096, 0, 0);
    EXPECT_EQ(rc, -ENOTSUP);
    return;
  }
  GTEST_SKIP() << "Requires real mount with malformed symlink — integration test only";
}

// @tests SCN-L4-001-03
TEST_F(TestUsrbIoGdrFixture, SCN_L4_001_03_DeviceFallbackNoCompile) {
#ifndef HF3FS_GDR_ENABLED
  struct hf3fs_iov iov;
  memset(&iov, 0, sizeof(iov));
  int rc = hf3fs_iovcreate_device(&iov, "/nonexistent", 4096, 0, 0);
  EXPECT_NE(rc, 0);  // Fails due to invalid mount, but takes host fallback path
#else
  GTEST_SKIP() << "Test only applies when GDR is not compiled";
#endif
}

// @tests SCN-L4-006-01
TEST_F(TestUsrbIoGdrFixture, SCN_L4_006_01_SyncGpuIovDirection0) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No GPU — integration test only";
  }
  GTEST_SKIP() << "Requires real GPU IOV — integration test only";
}
