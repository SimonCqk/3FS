FROM hub.docker.alibaba-inc.com/tre-ai-infra/3fs-build-base:ubuntu-22.04
COPY patches/eic-sdk_1.3.7.cuda12.ppu.202504161451_all.deb /tmp/eic-sdk_1.3.7.cuda12.ppu.202504161451_all.deb
COPY patches/unicm_1.7.1-2_amd64.deb /tmp/unicm_1.7.1-2_amd64.deb
RUN dpkg -i /tmp/eic-sdk_1.3.7.cuda12.ppu.202504161451_all.deb
RUN /opt/eic-sdk/install.bin
RUN dpkg -i /tmp/unicm_1.7.1-2_amd64.deb