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
DISK_NAME=$1
TAR_NAME=$2

sudo modprobe nbd max_part=10
sudo qemu-nbd -c /dev/nbd0 ${DISK_NAME}
sudo mount /dev/nbd0p3 ${ALPINE_BASE_MNT}/
cd mnt-alpine-tf-base/
sudo tar -cpzf ${TAR_NAME} --exclude=boot* --exclude=tmp* --exclude=lib/firmware/* *
cd ..
sudo umount ${ALPINE_BASE_MNT}
sudo qemu-nbd -d /dev/nbd0
