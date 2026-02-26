FROM ubuntu:22.04
COPY ./build/bin /opt/3fs/bin
COPY ./build/src/lib/api /opt/3fs/api
COPY ./build/third_party/jemalloc/lib/libjemalloc.so.2 /lib/x86_64-linux-gnu/
COPY ./configs /opt/3fs/etc
COPY ./deploy /opt/3fs/deploy
COPY ./patches/entrypoint.sh /opt/3fs/bin/3fs-entrypoint.sh
RUN chmod +x /opt/3fs/bin/3fs-entrypoint.sh
WORKDIR /opt/3fs/bin