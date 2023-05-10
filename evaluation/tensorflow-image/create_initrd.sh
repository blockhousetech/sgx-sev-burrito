#!/bin/bash
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
PYTHON_SCRIPT=$1
source ../../.env

SPATH=$(readlink -f "$0")
BASEDIR=$(dirname "$SPATH")


# Creating docker image
docker build --tag tf-image --build-arg  PYTHON_SCRIPT=${PYTHON_SCRIPT} .
docker save tf-image > tf-image.tar


# Adding docker image to rootfs and compressing it into a tar.gz
sudo qemu-nbd -c /dev/nbd0 ${ALPINE_BASE_IMAGE} 
sudo mount /dev/nbd0p3 ${ALPINE_BASE_MNT}
echo ${ALPINE_BASE_MNT}
sudo cp tf-image.tar ${ALPINE_BASE_MNT}/root/tf-image.tar
cd ${ALPINE_BASE_MNT}
sudo  tar -cpzf ${ALPINE_BASE_ROOTFS} --exclude=boot* --exclude=tmp* --exclude=lib/firmware/* *
cd ..
sudo umount ${ALPINE_BASE_MNT} 
sudo qemu-nbd -d /dev/nbd0

# creating initrd with above rootfs
cd ${BASEDIR}/../../sev_vm_initrd_build/
sudo ./update-initramfs ${ALPINE_BASE_INITRAMFS} ${ALPINE_BASE_ROOTFS} initramfs-init initramfs-alpine-tf-base
sudo cp initramfs-alpine-tf-base ../evaluation/tensorflow-image/${PYTHON_SCRIPT}_initramfs

cd ../evaluation/tensorflow-image
