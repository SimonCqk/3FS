FROM hub.docker.alibaba-inc.com/tre-ai-infra/3fs:v0.1.1
COPY patches/entrypoint.sh /opt/3fs/bin/3fs-entrypoint.sh
RUN chmod +x /opt/3fs/bin/3fs-entrypoint.sh
RUN apt-get update && apt install -y fio
WORKDIR /opt/3fs/bin