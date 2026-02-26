#include "IovTable.h"

#include <folly/experimental/coro/BlockingWait.h>

#include "IoRing.h"
#include "fbs/meta/Common.h"
#ifdef HF3FS_GDR_ENABLED
#include "common/net/ib/AcceleratorMemory.h"
#endif

namespace hf3fs::fuse {

using hf3fs::lib::IorAttrs;

const Path linkPref = "/dev/shm";

#ifdef HF3FS_GDR_ENABLED
namespace {
// Parse GDR URI: gdr://v1/device/{id}/size/{size}/ipc/{hex}
// Returns false if the target is not a valid GDR URI.
struct ParsedGdrTarget {
  int deviceId = -1;
  size_t size = 0;
  uint8_t ipcHandle[64] = {};
};

bool decodeHexBytes(const std::string &hex, uint8_t *out, size_t outLen) {
  if (hex.size() != outLen * 2) return false;
  for (size_t i = 0; i < outLen; ++i) {
    auto toNibble = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return -1;
    };
    int hi = toNibble(hex[2 * i]);
    int lo = toNibble(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<uint8_t>((hi << 4) | lo);
  }
  return true;
}

bool parseGdrTarget(const std::string &target, ParsedGdrTarget &out) {
  int deviceId = -1;
  unsigned long long size = 0;
  char ipcHex[129] = {};
  if (std::sscanf(target.c_str(),
                  "gdr://v1/device/%d/size/%llu/ipc/%128[0-9a-fA-F]",
                  &deviceId, &size, ipcHex) == 3) {
    auto prefix = fmt::format("gdr://v1/device/{}/size/{}/ipc/", deviceId, size);
    if (target.rfind(prefix, 0) != 0) return false;
    std::string encoded = target.substr(prefix.size());
    if (!decodeHexBytes(encoded, out.ipcHandle, 64)) return false;
    out.deviceId = deviceId;
    out.size = static_cast<size_t>(size);
    return true;
  }
  return false;
}
}  // namespace
#endif

void IovTable::init(const Path &mount, int cap) {
  mountName = mount.native();
  iovs = std::make_unique<AtomicSharedPtrTable<lib::ShmBuf>>(cap);
}

struct IovAttrs {
  Uuid id;
  size_t blockSize = 0;
  bool isIoRing = false;
  bool forRead = true;
  int ioDepth = 0;
  std::optional<IorAttrs> iora;
  bool isGdr = false;
  int gpuDeviceId = -1;
};

static Result<IovAttrs> parseKey(const char *key) {
  IovAttrs iova;

  std::vector<std::string> fnParts;
  folly::split('.', key, fnParts);

  auto idRes = Uuid::fromHexString(fnParts[0]);
  RETURN_ON_ERROR(idRes);
  iova.id = *idRes;

  for (size_t i = 1; i < fnParts.size(); ++i) {
    auto dec = fnParts[i];
    switch (dec[0]) {
      case 'b': {  // block size
        auto i = atoll(dec.c_str() + 1);
        if (i <= 0) {
          return makeError(StatusCode::kInvalidArg, "invalid block size set in shm key");
        }
        iova.blockSize = (size_t)i;
        break;
      }

      case 'r':
      case 'w': {  // is io ring
        auto i = atoll(dec.c_str() + 1);
        iova.isIoRing = true;
        iova.forRead = dec[0] == 'r';
        iova.ioDepth = i;
        break;
      }

      case 't': {
        if (!iova.iora) {
          iova.iora = IorAttrs{};
        }
        auto i = atoi(dec.c_str() + 1);
        if (i < 0) {
          return makeError(StatusCode::kInvalidArg, "invalid io job check timeout {}", dec.c_str() + 1);
        }
        iova.iora->timeout = Duration(std::chrono::nanoseconds((uint64_t)i * 1000000));
        break;
      }

      case 'f': {
        if (!iova.iora) {
          iova.iora = IorAttrs{};
        }
        char *ep;
        auto i = strtoull(dec.c_str() + 1, &ep, 2);
        if (*ep != 0 || i < 0) {
          return makeError(StatusCode::kInvalidArg, "invalid io exec flags {}", dec.c_str() + 1);
        }
        iova.iora->flags = i;
        break;
      }

      case 'p':  // should be io ring, priority
        if (!iova.iora) {
          iova.iora = IorAttrs{};
        }
        switch (dec.c_str()[1]) {
          case 'l':
            iova.iora->priority = 2;
            break;
          case 'h':
            iova.iora->priority = 0;
            break;
          case 'n':
          case '\0':
            iova.iora->priority = 1;
            break;
          default:
            return makeError(StatusCode::kInvalidArg, "invalid priority set in shm key");
        }
        break;

      case 'g':  // gdr marker (e.g. ".gdr")
        if (dec == "gdr") {
          iova.isGdr = true;
        }
        break;

      case 'd': {  // gpu device id (e.g. ".d0", ".d1")
        auto devId = atoi(dec.c_str() + 1);
        if (devId < 0) {
          return makeError(StatusCode::kInvalidArg, "invalid gpu device id in key");
        }
        iova.gpuDeviceId = devId;
        break;
      }
    }
  }

  if (!iova.isIoRing && iova.iora) {
    return makeError(StatusCode::kInvalidArg, "ioring attrs set for non-ioring");
  }

  return iova;
}

constexpr int iovIidStart = meta::InodeId::iovIidStart;

std::optional<int> IovTable::iovDesc(meta::InodeId iid) {
  auto iidn = (ssize_t)iid.u64();
  auto diid = (ssize_t)meta::InodeId::iovDir().u64();
  if (iidn >= 0 || iidn > diid - iovIidStart || iidn < diid - std::numeric_limits<int>::max()) {
    return std::nullopt;
  }
  return diid - iidn - iovIidStart;
}

Result<std::pair<meta::Inode, std::shared_ptr<lib::ShmBuf>>> IovTable::addIov(const char *key,
                                                                              const Path &shmPath,
                                                                              pid_t pid,
                                                                              const meta::UserInfo &ui,
                                                                              folly::Executor::KeepAlive<> exec,
                                                                              storage::client::StorageClient &sc) {
  static monitor::DistributionRecorder mapTimesCount("fuse.iov.times", monitor::TagSet{{"mount_name", mountName}});
  static monitor::DistributionRecorder mapBytesDist("fuse.iov.bytes", monitor::TagSet{{"mount_name", mountName}});
  static monitor::CountRecorder shmSizeCount("fuse.iov.total_bytes", monitor::TagSet{{"mount_name", mountName}}, false);
  static monitor::LatencyRecorder allocLatency("fuse.iov.latency.map", monitor::TagSet{{"mount_name", mountName}});
  static monitor::DistributionRecorder ibRegBytesDist("fuse.iov.bytes.ib_reg",
                                                      monitor::TagSet{{"mount_name", mountName}});
  static monitor::LatencyRecorder ibRegLatency("fuse.iov.latency.ib_reg", monitor::TagSet{{"mount_name", mountName}});

  auto iovaRes = parseKey(key);
  RETURN_ON_ERROR(iovaRes);

#ifndef HF3FS_GDR_ENABLED
  if (iovaRes->isGdr) {
    return makeError(StatusCode::kInvalidArg, "GDR not enabled in this build");
  }
#endif

#ifdef HF3FS_GDR_ENABLED
  // GDR path: shmPath is a gdr:// URI, not a filesystem path
  if (iovaRes->isGdr) {
    ParsedGdrTarget gdrTarget;
    if (!parseGdrTarget(shmPath.native(), gdrTarget)) {
      return makeError(StatusCode::kInvalidArg, "failed to parse GDR target URI");
    }

    // parseGdrTarget success guarantees a valid IPC handle
    lib::GpuIpcHandle ipcHandle;
    std::memcpy(ipcHandle.data, gdrTarget.ipcHandle, 64);
    ipcHandle.valid = true;

    // Import the GPU memory via IPC handle
    auto gpuShm = std::make_shared<lib::GpuShmBuf>(
        ipcHandle, gdrTarget.size, gdrTarget.deviceId, iovaRes->id);

    if (!gpuShm->devicePtr) {
      return makeError(StatusCode::kInvalidArg, "failed to import GPU memory via IPC handle");
    }

    gpuShm->key = key;
    gpuShm->user = ui.uid;
    gpuShm->pid = pid;

    // Allocate iov descriptor slot
    auto iovdRes = iovs->alloc();
    if (!iovdRes) {
      return makeError(ClientAgentCode::kTooManyOpenFiles, "too many iovs allocated");
    }
    auto iovd = *iovdRes;
    bool dealloc = true;
    SCOPE_EXIT {
      if (dealloc) {
        iovs->dealloc(iovd);
      }
    };

    // We store a null ShmBuf in the slot (GPU buffers are tracked via gpuShmsById)
    // but we still need the slot for iov descriptor numbering
    iovs->table[iovd].store(nullptr);

    // Register GPU memory for RDMA I/O
    auto recordMetrics = []() {};
    folly::coro::blockingWait(gpuShm->registerForIO(exec, sc, std::move(recordMetrics)));

    {
      std::unique_lock lock(iovdLock_);
      iovds_[key] = iovd;
    }

    {
      std::lock_guard lock(gpuShmLock);
      gpuShmsById[iovaRes->id] = gpuShm;
      gpuIovMetaByIovd[iovd] = GpuIovMeta{std::string(key), shmPath, ui.uid, ui.gid};
    }

    // For GPU iovs, we return the GDR URI as the symlink target
    auto inode = meta::Inode{
        meta::InodeId::iov(iovd),
        meta::InodeData{meta::Symlink{shmPath}, meta::Acl{ui.uid, ui.gid, meta::Permission(0400)}}};

    dealloc = false;
    return std::make_pair(inode, std::shared_ptr<lib::ShmBuf>());
  }
#endif

  Path shmOpenPath("/");
  shmOpenPath /= shmPath.lexically_relative(linkPref);

  struct stat st;
  if (stat(shmPath.c_str(), &st) == -1 || !S_ISREG(st.st_mode)) {
    return makeError(StatusCode::kInvalidArg, "failed to stat shm path or it's not a regular file");
  }

  if (iovaRes->blockSize > (size_t)st.st_size) {
    return makeError(StatusCode::kInvalidArg, "invalid block size set in shm key");
  } else if (iovaRes->isIoRing && iovaRes->ioDepth > IoRing::ioRingEntries((size_t)st.st_size)) {
    return makeError(StatusCode::kInvalidArg, "invalid io batch size set in shm key");
  }

  while (true) {
    auto iovdRes = iovs->alloc();
    if (!iovdRes) {
      return makeError(ClientAgentCode::kTooManyOpenFiles, "too many iovs allocated");
    }
    auto iovd = *iovdRes;
    bool dealloc = true;
    SCOPE_EXIT {
      if (dealloc) {
        iovs->dealloc(iovd);
      }
    };

    auto start = SteadyClock::now();
    auto uids = std::to_string(ui.uid.toUnderType());

    std::shared_ptr<lib::ShmBuf> shm;
    try {
      shm.reset(
          new lib::ShmBuf(shmOpenPath, 0, st.st_size, iovaRes->blockSize, iovaRes->id),
          [uids,
           &shmSizeCount = shmSizeCount,
           &mapTimesCount = mapTimesCount,
           &mapBytesDist = mapBytesDist,
           &allocLatency = allocLatency,
           &ibRegLatency = ibRegLatency](auto p) {
            auto start = SteadyClock::now();
            folly::coro::blockingWait(p->deregisterForIO());
            auto now = SteadyClock::now();
            ibRegLatency.addSample(now - start, monitor::TagSet{{"instance", "dereg"}, {"uid", uids}});

            start = now;
            p->unmapBuf();
            allocLatency.addSample(SteadyClock::now() - start, monitor::TagSet{{"instance", "free"}, {"uid", uids}});

            mapTimesCount.addSample(1, monitor::TagSet{{"instance", "free"}, {"uid", uids}});
            mapBytesDist.addSample(p->size, monitor::TagSet{{"instance", "free"}, {"uid", uids}});
            shmSizeCount.addSample(-p->size);

            delete p;
          });
    } catch (const std::runtime_error &e) {
      return makeError(ClientAgentCode::kIovShmFail, std::string("failed to open/map shm for iov ") + e.what());
    }

    allocLatency.addSample(SteadyClock::now() - start, monitor::TagSet{{"instance", "alloc"}, {"uid", uids}});
    mapTimesCount.addSample(1, monitor::TagSet{{"instance", "alloc"}, {"uid", uids}});
    mapBytesDist.addSample(shm->size, monitor::TagSet{{"instance", "alloc"}, {"uid", uids}});
    shmSizeCount.addSample(shm->size, monitor::TagSet{{"uid", uids}});

    shm->key = key;
    shm->user = ui.uid;
    shm->pid = pid;
    shm->isIoRing = iovaRes->isIoRing;
    shm->forRead = iovaRes->forRead;
    shm->ioDepth = iovaRes->ioDepth;
    shm->iora = iovaRes->iora;

    // the idx should be reserved by us
    iovs->table[iovd].store(shm);

    start = SteadyClock::now();
    auto recordMetrics = [blockSize = shm->blockSize, start, uids]() mutable {
      ibRegBytesDist.addSample(blockSize, monitor::TagSet{{"instance", "reg"}, {"uid", uids}});
      ibRegLatency.addSample(SteadyClock::now() - start, monitor::TagSet{{"instance", "reg"}, {"uid", uids}});
    };

    if (!iovaRes->isIoRing) {  // io ring bufs don't need to be registered for ib io
      folly::coro::blockingWait(shm->registerForIO(exec, sc, recordMetrics));
    }

    {
      std::unique_lock lock(iovdLock_);
      iovds_[key] = iovd;
    }

    {
      std::unique_lock lock(shmLock);
      shmsById[iovaRes->id] = iovd;
    }

    auto statRes = statIov(iovd, ui);
    RETURN_ON_ERROR(statRes);

    dealloc = false;
    return std::make_pair(*statRes, iovaRes->isIoRing ? shm : std::shared_ptr<lib::ShmBuf>());
  }
}

Result<std::shared_ptr<lib::ShmBuf>> IovTable::rmIov(const char *key, const meta::UserInfo &ui) {
  auto res = lookupIov(key, ui);
  RETURN_ON_ERROR(res);

  {
    std::unique_lock lock(iovdLock_);
    iovds_.erase(key);
  }

  auto parseRes = parseKey(key);

#ifdef HF3FS_GDR_ENABLED
  if (parseRes && parseRes->isGdr) {
    // GPU iov: clean up from gpuShmsById and gpuIovMetaByIovd
    auto iovd = iovDesc(res->id);
    {
      std::lock_guard lock(gpuShmLock);
      gpuShmsById.erase(parseRes->id);
      if (iovd) {
        gpuIovMetaByIovd.erase(*iovd);
      }
    }

    if (iovd) {
      // Must use dealloc() directly — not remove().
      // GPU slots store nullptr in iovs->table, and remove() early-returns
      // on null slots without calling dealloc(), leaking the descriptor.
      iovs->dealloc(*iovd);
    }

    return std::shared_ptr<lib::ShmBuf>();
  }
#endif

  {
    std::unique_lock lock(shmLock);
    if (parseRes) {
      shmsById.erase(parseRes->id);
    }
  }

  auto iovd = iovDesc(res->id);
  auto shm = iovs->table[*iovd].load();
  iovs->remove(*iovd);

  return shm;
}

Result<meta::Inode> IovTable::statIov(int iovd, const meta::UserInfo &ui) {
  if (iovd < 0 || iovd >= (int)iovs->table.size()) {
    return makeError(MetaCode::kNotFound, "invalid iov desc");
  }

  auto shm = iovs->table[iovd].load();
  if (!shm) {
#ifdef HF3FS_GDR_ENABLED
    // Check if this is a GPU iov
    std::lock_guard lock(gpuShmLock);
    auto git = gpuIovMetaByIovd.find(iovd);
    if (git != gpuIovMetaByIovd.end()) {
      auto &meta = git->second;
      if (meta.user != ui.uid) {
        XLOGF(ERR, "statting user {} gpu iov belongs to {}", ui.uid, meta.user);
        return makeError(MetaCode::kNoPermission, "iov not for user");
      }
      return meta::Inode{
          meta::InodeId::iov(iovd),
          meta::InodeData{meta::Symlink{meta.target}, meta::Acl{ui.uid, meta.gid, meta::Permission(0400)}}};
    }
#endif
    return makeError(MetaCode::kNotFound,
                     fmt::format("iov desc {} not found, next avail {}", iovd, iovs->slots.nextAvail.load()));
  }

  if (shm->user != ui.uid) {
    XLOGF(ERR, "statting user {} iov belongs to {}", ui.uid, shm->user);
    return makeError(MetaCode::kNoPermission, "iov not for user");
  }

  return meta::Inode{
      meta::InodeId::iov(iovd),
      meta::InodeData{meta::Symlink{linkPref / shm->path}, meta::Acl{ui.uid, ui.gid, meta::Permission(0400)}}};
}

Result<meta::Inode> IovTable::lookupIov(const char *key, const meta::UserInfo &ui) {
  int iovd = -1;
  {
    std::shared_lock lock(iovdLock_);
    auto it = iovds_.find(key);
    if (it == iovds_.end()) {
      return makeError(MetaCode::kNotFound, std::string("iov key not found ") + key);
    } else {
      iovd = it->second;
    }
  }

  return statIov(iovd, ui);
}

std::pair<std::shared_ptr<std::vector<meta::DirEntry>>, std::shared_ptr<std::vector<std::optional<meta::Inode>>>>
IovTable::listIovs(const meta::UserInfo &ui) {
  meta::DirEntry de{meta::InodeId::iovDir(), ""};

  auto n = iovs->slots.nextAvail.load();
  std::vector<meta::DirEntry> des;
  std::vector<std::optional<meta::Inode>> ins;
  des.reserve(n + 3);
  ins.reserve(n + 3);

  for (int prio = 0; prio <= 2; ++prio) {
    de.name = IoRingTable::semName(prio);
    des.emplace_back(de);

    auto inode = IoRingTable::lookupSem(prio);
    ins.emplace_back(std::move(inode));
  }

  meta::Acl acl{meta::Uid{ui.uid}, meta::Gid{ui.gid}, meta::Permission{0400}};

#ifdef HF3FS_GDR_ENABLED
  // Snapshot GPU metadata under a single lock before iterating
  robin_hood::unordered_map<int, GpuIovMeta> gpuMetaSnapshot;
  {
    std::lock_guard lock(gpuShmLock);
    gpuMetaSnapshot = gpuIovMetaByIovd;
  }
#endif

  for (int i = 0; i < n; ++i) {
    auto iov = iovs->table[i].load();
    if (iov) {
      if (iov->user != ui.uid) {
        continue;
      }
      de.name = iov->key;
      des.emplace_back(de);
      ins.emplace_back(
          meta::Inode{meta::InodeId{meta::InodeId::iov(i)}, meta::InodeData{meta::Symlink{linkPref / iov->path}, acl}});
      continue;
    }

#ifdef HF3FS_GDR_ENABLED
    // Check for GPU iov from snapshot
    auto git = gpuMetaSnapshot.find(i);
    if (git != gpuMetaSnapshot.end() && git->second.user == ui.uid) {
      de.name = git->second.key;
      des.emplace_back(de);
      ins.emplace_back(meta::Inode{
          meta::InodeId{meta::InodeId::iov(i)},
          meta::InodeData{meta::Symlink{git->second.target},
                          meta::Acl{git->second.user, git->second.gid, meta::Permission{0400}}}});
    }
#endif
  }

  return std::make_pair(std::make_shared<std::vector<meta::DirEntry>>(std::move(des)),
                        std::make_shared<std::vector<std::optional<meta::Inode>>>(std::move(ins)));
}
}  // namespace hf3fs::fuse
