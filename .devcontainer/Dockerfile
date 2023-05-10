##
## Burrito
## Copyright (C) 2023 The Blockhouse Technology Limited (TBTL)
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU Affero General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Affero General Public License for more details.
##
## You should have received a copy of the GNU Affero General Public License
## along with this program. If not, see <https://www.gnu.org/licenses/>.
##
FROM ubuntu:18.04

RUN useradd -s /bin/bash -m dev
ENV HOME /home/dev
RUN apt update -y && \
    apt upgrade -y

## GIT
RUN apt install git -y
RUN apt install openssh-client -y

## GDB
RUN apt install gdb -y

## SGX - assumes that the EPID driver has been installed in host
# Dependencies
RUN apt install libssl-dev -y
RUN apt install build-essential -y
RUN apt install libprotobuf-dev -y
RUN apt install libcurl4-openssl-dev -y
RUN apt install wget -y
RUN apt install unzip -y

RUN apt update -y

# PSW
RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu bionic main' | tee /etc/apt/sources.list.d/intel-sgx.list
RUN wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN apt update -y

RUN apt install libsgx-urts -y
RUN apt install libsgx-epid -y
RUN apt install libsgx-quote-ex -y

# Debug symbols for libsgx-urts
RUN apt install libsgx-urts-dbgsym -y

# SGX SDK
RUN cd ${HOME} \ 
    && wget https://download.01.org/intel-sgx/sgx-linux/2.17/distro/ubuntu18.04-server/sgx_linux_x64_sdk_2.17.100.3.bin \
    && chmod +x sgx_linux_x64_sdk_2.17.100.3.bin \
    && echo "yes" | ./sgx_linux_x64_sdk_2.17.100.3.bin

# SGX SSL
ARG SGXSSL_FILENAME=c3a3a3010965a28d1eaa350b087c5159a434d2bf
RUN cd ${HOME} && wget https://github.com/intel/intel-sgx-ssl/archive/${SGXSSL_FILENAME}.zip
RUN cd ${HOME} && unzip -qq ${SGXSSL_FILENAME}.zip
RUN mv ${HOME}/intel-sgx-ssl-${SGXSSL_FILENAME} ${HOME}/sgxssl
RUN rm ${HOME}/${SGXSSL_FILENAME}.zip

ARG OPENSSL_FILE=openssl-1.1.1o.tar.gz
RUN wget https://www.openssl.org/source/${OPENSSL_FILE} -P ${HOME}/sgxssl/openssl_source

RUN . /home/dev/sgxsdk/environment && cd ${HOME}/sgxssl/Linux \
    && make clean sgxssl_no_mitigation

# SGX DCAP
RUN apt install -y libsgx-enclave-common-dev libsgx-dcap-default-qpl libsgx-dcap-default-qpl-dev libsgx-dcap-ql libsgx-dcap-ql-dev

# GRPC
## cmake pre-requisite
RUN mkdir -p ${HOME}/cmake
RUN cd ${HOME} && wget -q -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-Linux-x86_64.sh
RUN cd ${HOME} && sh cmake-linux.sh -- --skip-license --prefix=${HOME}/cmake
RUN cd ${HOME} && rm cmake-linux.sh
ENV PATH="${PATH}:${HOME}/cmake/bin"
## end cmake

ARG MY_INSTALL_DIR=/usr/local
RUN apt install -y build-essential autoconf libtool pkg-config
RUN cd ${HOME} && git clone --recurse-submodules -b v1.46.3 --depth 1 --shallow-submodules https://github.com/grpc/grpc
RUN cd ${HOME}/grpc && mkdir -p cmake/build
RUN bash -c "cd ${HOME}/grpc && \
        pushd cmake/build && \ 
        cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${MY_INSTALL_DIR} ../.. && \
        make -j4 && \ 
        make install && \
        popd"

RUN apt install -y libsgx-dcap-quote-verify-dev libsgx-dcap-quote-verify

# VSCode extensions will be cached in a named volume
# for faster rebuilds
RUN mkdir -p \
    ${HOME}/.vscode-server/extensions \
    ${HOME}/.vscode-server-insiders/extensions

# rpc port
EXPOSE 50051

RUN mkdir /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN su - dev -c 'echo "ssh-add -k /run/secrets/id_rsa" >> /home/dev/.bashrc'
RUN su - dev -c "echo 'source /home/dev/sgxsdk/environment' >> /home/dev/.bashrc"
RUN echo "Done"

RUN apt install -y python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install grpcio
RUN python3 -m pip install grpcio-tools

# KEEP THE DOCKER RUNNING
CMD tail -f /dev/null