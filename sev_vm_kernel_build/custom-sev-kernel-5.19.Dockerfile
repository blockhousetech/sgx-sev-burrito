# Dockerfile for customer sev kernel
FROM alpine:3.14 AS build-stage
RUN apk add alpine-sdk bison flex linux-headers openssl-dev elfutils-dev
RUN abuild-keygen -na
RUN git clone https://gitlab.alpinelinux.org/alpine/aports
WORKDIR /aports/main/linux-lts
RUN git checkout 3.14-stable
RUN rm *
COPY config-virt.x86_64 ./
COPY APKBUILD ./
RUN abuild -F checksum
RUN abuild -F -r

# Export stage: copy only the files that are needed.
FROM scratch
COPY --from=build-stage /root/packages/main/x86_64/linux-virt-5.19-r0.apk /
