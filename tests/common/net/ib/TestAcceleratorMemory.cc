#include <gtest/gtest.h>

#include "SetupAccelerator.h"
#include "common/net/ib/AcceleratorMemory.h"
#include "common/net/ib/MemoryTypes.h"

namespace hf3fs::net {

class TestAcceleratorMemory : public test::SetupAccelerator {};

TEST_F(TestAcceleratorMemory, SkipWithoutGpu) {
  if (!test::SetupAccelerator::hasGpuSupport()) {
    GTEST_SKIP() << "GDR not available, skipping GPU tests";
  }
  SUCCEED();
}

TEST_F(TestAcceleratorMemory, GdrManagerAvailability) {
  auto& manager = GDRManager::instance();
  bool available = manager.isAvailable();
  
  if (available) {
    EXPECT_GT(manager.getGpuDevices().size(), 0u);
  }
  
  SUCCEED();
}

TEST_F(TestAcceleratorMemory, HostPointerDetection) {
  size_t size = 4096;
  void* host_ptr = malloc(size);
  ASSERT_NE(host_ptr, nullptr);

  MemoryType mem_type = detectMemoryType(host_ptr);
  EXPECT_EQ(mem_type, MemoryType::Host);

  free(host_ptr);
}

TEST_F(TestAcceleratorMemory, MemoryTypeEnumValues) {
  EXPECT_NE(MemoryType::Host, MemoryType::Device);
  EXPECT_NE(MemoryType::Host, MemoryType::Managed);
  EXPECT_NE(MemoryType::Device, MemoryType::Managed);
  EXPECT_EQ(static_cast<int>(MemoryType::Host), 0);
}

TEST_F(TestAcceleratorMemory, DeviceVendorEnumValues) {
  EXPECT_NE(DeviceVendor::None, DeviceVendor::NVIDIA);
  EXPECT_NE(DeviceVendor::None, DeviceVendor::AMD);
  EXPECT_NE(DeviceVendor::None, DeviceVendor::Intel);
  EXPECT_EQ(static_cast<int>(DeviceVendor::None), 0);
}

#ifdef HF3FS_GDR_ENABLED
TEST_F(TestAcceleratorMemory, GpuMemoryDescriptorStructure) {
  if (!test::SetupAccelerator::hasGpuSupport()) {
    GTEST_SKIP() << "GDR not available, skipping GPU tests";
  }
  
  auto& manager = GDRManager::instance();
  ASSERT_TRUE(manager.isAvailable());
  
  AcceleratorMemoryDescriptor desc;
  desc.size = 4096;
  desc.deviceId = 0;
  
  EXPECT_EQ(desc.size, 4096u);
  EXPECT_EQ(desc.deviceId, 0);
  EXPECT_EQ(desc.devicePtr, nullptr);
  EXPECT_FALSE(desc.ipcHandle.valid);
}
#endif

TEST_F(TestAcceleratorMemory, GdrConfigDefaults) {
  GDRConfig config;
  
  EXPECT_FALSE(config.enabled());
  EXPECT_GT(config.max_cached_regions(), 0u);
  EXPECT_EQ(config.required_alignment(), 256u);
}

}  // namespace hf3fs::net
