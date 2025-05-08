# FROM registry.cn-hangzhou.aliyuncs.com/qiukai-dev/3fs-dev-base:ubuntu-22.04
FROM ubuntu:22.04
# COPY . /app/3FS
# FROM ${BASE_IMAGE}
# COPY --from=builder /app/3FS/build/bin /opt/3fs/bin
# COPY --from=builder /app/3FS/configs /opt/3fs/etc
# COPY --from=builder /app/3FS/deploy /opt/3fs/deploy
# COPY --from=builder /app/3FS/build/third_party/jemalloc/lib/libjemalloc.so.2 /lib/x86_64-linux-gnu/
COPY ./build/bin /opt/3fs/bin
COPY ./build/src/lib/api /opt/3fs/api
COPY ./build/third_party/jemalloc/lib/libjemalloc.so.2 /lib/x86_64-linux-gnu/
COPY ./configs /opt/3fs/etc
COPY ./deploy /opt/3fs/deploy
#COPY ./build/third_party/jemalloc/lib/libjemalloc.so.2 /lib/x86_64-linux-gnu/
#COPY ./build /3fs-build-output
COPY ./patches/entrypoint.sh /opt/3fs/bin/3fs-entrypoint.sh
RUN chmod +x /opt/3fs/bin/3fs-entrypoint.sh
WORKDIR /opt/3fs/bin