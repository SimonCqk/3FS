#include <gtest/gtest.h>

#include "SetupAccelerator.h"
#include "common/net/ib/GpuMemory.h"

namespace hf3fs::net {

class TestAcceleratorMemory : public test::SetupAccelerator {};

TEST_F(TestAcceleratorMemory, SkipWithoutGpu) {
  if (!test::SetupAccelerator::hasGpuSupport()) {
    GTEST_SKIP() << "GDR not available, skipping GPU tests";
  }
  SUCCEED();
}

}  // namespace hf3fs::net
