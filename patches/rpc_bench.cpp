#include <chrono>
#include <folly/Random.h>
#include <folly/Unit.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/GtestHelpers.h>
#include <folly/experimental/coro/Invoke.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "common/net/Client.h"
#include "common/net/Listener.h"
#include "common/net/Server.h"
#include "common/net/ib/IBDevice.h"
#include "common/net/sync/Client.h"
#include "common/serde/ClientContext.h"
#include "common/utils/Address.h"
#include "common/utils/Coroutine.h"
#include "tests/GtestHelpers.h"
#include "tests/common/net/Echo.h"
#include "tests/common/net/ib/SetupIB.h"
#include <ylt/metric/summary.hpp>

DEFINE_uint32(req_len, 13, "request data len");
DEFINE_uint32(par, 10, "parallel");
DEFINE_uint32(port, 0, "port");
DEFINE_string(host, "10.0.0.42:9004", "host");
DEFINE_uint32(tcp, 1, "enable tcp, otherwise rdma");
DEFINE_uint32(dur, 15, "test duration");

using namespace hf3fs;
using namespace hf3fs::net;
using namespace hf3fs::net::test;

std::atomic<uint64_t> g_count = 0;
ylt::metric::summary_t g_latency{"Latency(us) of rpc call", "help",
                                 std::vector{0.5, 0.9, 0.95, 0.99},
                                 std::chrono::seconds{60}};

class EchoServiceImpl : public serde::ServiceWrapper<EchoServiceImpl, Echo> {
 public:
  CoTryTask<EchoRsp> echo(serde::CallContext &ctx, const EchoReq &req) {
    EchoRsp rsp;
    rsp.val = "hello for test";
    co_return rsp;
  }

  CoTryTask<HelloRsp> hello(serde::CallContext &ctx, const HelloReq &req) {
    HelloRsp rsp;
    rsp.val = "Hello, " + req.val;
    rsp.idx = ++idx_;
    co_return rsp;
  }

  CoTryTask<HelloRsp> fail(serde::CallContext &ctx, const HelloReq &) {
    fmt::print("Request from {}\n", ctx.transport()->describe());
    co_return makeError(RPCCode::kInvalidMessageType, "failed");
  }

 private:
  uint32_t idx_ = 0;
};

int main(int argc, char **argv) {
  folly::Init init(&argc, &argv);
  if(FLAGS_tcp == 0) {
  static IBConfig config;
  config.set_allow_unknown_zone(true);
  config.set_device_filter({"mlx5_bond_0"});
  auto ib = IBManager::start(config);
  std::cout << "is empty " << IBDevice::all().empty() << "\n";
  XLOGF_IF(FATAL, ib.hasError(), "IBManager start failed, result {}", ib.error());
  assert(IBDevice::all().empty());
  }

  if (FLAGS_port > 0) {
    hf3fs::net::Server::Config serverConfig;
    if(FLAGS_tcp == 1) {
      serverConfig.groups(0).set_network_type(Address::TCP);
    }else {
      serverConfig.groups(0).set_network_type(Address::RDMA);
    }
    serverConfig.groups(0).listener().set_listen_port(FLAGS_port);
    serverConfig.groups(0).io_worker().transport_pool().set_max_connections(2000);
    serverConfig.groups(0).io_worker().set_num_event_loop(std::thread::hardware_concurrency());
    serverConfig.groups(0).io_worker().set_read_write_tcp_in_event_thread(true);
    serverConfig.groups(0).io_worker().ibsocket().set_buf_size(2 * 1024 * 1024);
    serverConfig.thread_pool().set_num_io_threads(std::thread::hardware_concurrency());
    serverConfig.thread_pool().set_num_proc_threads(std::thread::hardware_concurrency());
    // serverConfig.thread_pool().set_num_bg_threads(std::thread::hardware_concurrency());
    std::cout << "start server, port " << FLAGS_port << ", enable tcp " << FLAGS_tcp  << "\n";
    hf3fs::net::Server server{serverConfig};
    server.addSerdeService(std::make_unique<EchoServiceImpl>());
    server.setup();
    server.start();

    std::string wait_quit;
    std::cin >> wait_quit;
  }
}