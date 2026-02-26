#include <boost/core/ignore_unused.hpp>

#ifdef HF3FS_GDR_ENABLED
#include <cuda_runtime.h>
#endif

#include "StorageClientImpl.h"
#include "StorageClientInMem.h"
#include "common/monitor/ScopedMetricsWriter.h"
#include "common/net/ib/RDMABuf.h"

namespace hf3fs::storage::client {

static monitor::CountRecorder iobuf_reg_success_ops{"storage_client.iobuf_reg.success_ops"};
static monitor::CountRecorder iobuf_reg_failed_ops{"storage_client.iobuf_reg.failed_ops"};
static monitor::LatencyRecorder iobuf_reg_latency{"storage_client.iobuf_reg.latency"};
static monitor::DistributionRecorder iobuf_reg_size{"storage_client.iobuf_reg.size"};

const StorageClient::Config StorageClient::kDefaultConfig;

std::shared_ptr<StorageClient> StorageClient::create(ClientId clientId,
                                                     const Config &config,
                                                     hf3fs::client::ICommonMgmtdClient &mgmtdClient) {
  const auto &trafficControl = config.traffic_control();

  if (trafficControl.max_concurrent_updates() > UpdateChannelAllocator::kMaxNumChannels) {
    XLOGF(CRITICAL,
          "Bad config: trafficControl.max_concurrent_updates {} > UpdateChannelAllocator::kMaxNumChannels {}",
          trafficControl.max_concurrent_updates(),
          UpdateChannelAllocator::kMaxNumChannels);
    return nullptr;
  }

  std::shared_ptr<StorageClient> client;

  if (config.implementation_type() == ImplementationType::RPC) {
    client = std::make_shared<StorageClientImpl>(clientId, config, mgmtdClient);
  } else if (config.implementation_type() == ImplementationType::InMem) {
    client = std::make_shared<StorageClientInMem>(clientId, config, mgmtdClient);
  }

  if (!client || !client->start()) {
    XLOGF(CRITICAL,
          "Failed to create and start storage client of type {}",
          magic_enum::enum_name(config.implementation_type()));
    client.reset();
  }

  return client;
}

ReadIO StorageClient::createReadIO(ChainId chainId,
                                   const ChunkId &chunkId,
                                   uint32_t offset,
                                   uint32_t length,
                                   uint8_t *data,
                                   IOBuffer *buffer,
                                   void *userCtx) {
  return ReadIO{chainId, chunkId, offset, length, data, buffer, userCtx};
}

WriteIO StorageClient::createWriteIO(ChainId chainId,
                                     const ChunkId &chunkId,
                                     uint32_t offset,
                                     uint32_t length,
                                     uint32_t chunkSize,
                                     uint8_t *data,
                                     IOBuffer *buffer,
                                     void *userCtx) {
  RequestId requestId(nextRequestId_.fetch_add(1));
  return WriteIO{requestId, chainId, chunkId, offset, length, chunkSize, data, buffer, userCtx};
}

QueryLastChunkOp StorageClient::createQueryOp(ChainId chainId,
                                              ChunkId chunkIdBegin,
                                              ChunkId chunkIdEnd,
                                              uint32_t maxNumChunkIdsToProcess,
                                              void *userCtx) {
  return QueryLastChunkOp{chainId, {chunkIdBegin, chunkIdEnd, maxNumChunkIdsToProcess}, userCtx};
}

RemoveChunksOp StorageClient::createRemoveOp(ChainId chainId,
                                             ChunkId chunkIdBegin,
                                             ChunkId chunkIdEnd,
                                             uint32_t maxNumChunkIdsToProcess,
                                             void *userCtx) {
  RequestId requestId(nextRequestId_.fetch_add(1));
  return RemoveChunksOp{requestId, chainId, {chunkIdBegin, chunkIdEnd, maxNumChunkIdsToProcess}, userCtx};
}

TruncateChunkOp StorageClient::createTruncateOp(ChainId chainId,
                                                const ChunkId &chunkId,
                                                uint32_t chunkLen,
                                                uint32_t chunkSize,
                                                bool onlyExtendChunk,
                                                void *userCtx) {
  RequestId requestId(nextRequestId_.fetch_add(1));
  return TruncateChunkOp(requestId, chainId, chunkId, chunkLen, chunkSize, onlyExtendChunk, userCtx);
}

Result<IOBuffer> StorageClient::registerIOBuffer(uint8_t *buf, size_t len) {
  monitor::ScopedLatencyWriter latencyWriter(iobuf_reg_latency);
  iobuf_reg_size.addSample(len);

  auto rdmabuf = hf3fs::net::RDMABuf::createFromUserBuffer(buf, len);

  if (rdmabuf.valid()) {
    iobuf_reg_success_ops.addSample(1);
    return IOBuffer{rdmabuf};
  } else {
    iobuf_reg_failed_ops.addSample(1);
    return makeError(StorageClientCode::kMemoryError);
  }
}

Result<IOBuffer> StorageClient::registerGpuIOBuffer(uint8_t *gpuPtr, size_t len) {
  monitor::ScopedLatencyWriter latencyWriter(iobuf_reg_latency);
  iobuf_reg_size.addSample(len);

#ifdef HF3FS_GDR_ENABLED
  int deviceId = 0;
  cudaPointerAttributes attrs;
  if (cudaPointerGetAttributes(&attrs, gpuPtr) == cudaSuccess
      && attrs.type == cudaMemoryTypeDevice) {
    deviceId = attrs.device;
  } else {
    cudaGetLastError();  // Clear CUDA error state
    XLOGF(WARN, "Could not detect GPU device for ptr {}, defaulting to 0", fmt::ptr(gpuPtr));
  }
  auto gpuBuf = hf3fs::net::RDMABufAccelerator::createFromGpuPointer(gpuPtr, len, deviceId);
  if (gpuBuf.valid()) {
    iobuf_reg_success_ops.addSample(1);
    return IOBuffer{hf3fs::net::RDMABufUnified(std::move(gpuBuf))};
  }
  iobuf_reg_failed_ops.addSample(1);
  return makeError(StorageClientCode::kMemoryError);
#else
  (void)gpuPtr;
  (void)len;
  iobuf_reg_failed_ops.addSample(1);
  return makeError(StorageClientCode::kMemoryError, "GPU IOBuffer requires HF3FS_GDR_ENABLED");
#endif
}

}  // namespace hf3fs::storage::client
