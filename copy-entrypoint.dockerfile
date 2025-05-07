FROM hub.docker.alibaba-inc.com/aone/tre-ai-infra/3fs:v0.1
COPY patches/entrypoint.sh /opt/3fs/bin/3fs-entrypoint.sh
RUN chmod +x /opt/3fs/bin/3fs-entrypoint.sh
WORKDIR /opt/3fs/bin