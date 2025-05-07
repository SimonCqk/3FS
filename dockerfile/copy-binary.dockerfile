FROM hub.docker.alibaba-inc.com/tre-ai-infra/3fs:v0.1
COPY patches/entrypoint.sh /opt/3fs/bin/3fs-entrypoint.sh
RUN chmod +x /opt/3fs/bin/3fs-entrypoint.sh
RUN apt-get update && apt install -y wget build-essential gawk bison alien rdma-core ibverbs-utils infiniband-diags perftest 
RUN wget https://scc-net.oss-cn-hangzhou.aliyuncs.com/nic-libs-mellanox-rdma-5.2-2.x86_64.rpm && alien -k nic-libs-mellanox-rdma-5.2-2.x86_64.rpm && dpkg -i --force-overwrite nic-libs-mellanox-rdma_5.2-2_amd64.deb
WORKDIR /opt/3fs/bin
# FROM registry.cn-hangzhou.aliyuncs.com/qiukai-dev/3fs:v0.1-20250507 AS builder
# FROM hub.docker.alibaba-inc.com/tre-ai-infra/demo-3fs:v0.6 as api-builder
# FROM hub.docker.alibaba-inc.com/tre-ai-infra/3fs-dev-base:ubuntu-22.04
# COPY --from=builder /opt/3fs/bin /opt/3fs/bin
# COPY --from=builder /opt/3fs/api /opt/3fs/api
# COPY --from=builder /opt/3fs/etc /opt/3fs/etc
# COPY --from=builder /opt/3fs/deploy /opt/3fs/deploy
# COPY --from=api-builder /lib/x86_64-linux-gnu/libjemalloc.so.2 /lib/x86_64-linux-gnu/libjemalloc.so.2
# COPY patches/entrypoint.sh /opt/3fs/etc/3fs-entrypoint.sh