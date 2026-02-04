#pragma once

#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "SetupIB.h"
#include "common/net/ib/GpuMemory.h"
#include "tests/GtestHelpers.h"

namespace hf3fs::net::test {

class SetupAccelerator : public SetupIB {
 public:
  static void SetUpTestSuite() {
    SetupIB::SetUpTestSuite();  // Initialize IB first
    // GDRManager is a singleton and initializes on demand
    // No explicit initialization needed here
  }

  static bool hasGpuSupport() {
    return GDRManager::instance().isAvailable();
  }
};

}  // namespace hf3fs::net::test
