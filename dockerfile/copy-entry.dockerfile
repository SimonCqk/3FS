FROM hub.docker.alibaba-inc.com/tre-ai-infra/3fs:v0.1
COPY patches/entrypoint.sh /opt/3fs/bin/3fs-entrypoint.sh
RUN chmod +x /opt/3fs/bin/3fs-entrypoint.sh
RUN apt update && apt install -y rdma-core ibverbs-utils infiniband-diags mlnx-tools
WORKDIR /opt/3fs/bin