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
source ../.env
PYTHON_SCRIPT=$1

cp ${PYTHON_SCRIPT} ../evaluation/tensorflow-image/${PYTHON_SCRIPT}
cd ../evaluation/tensorflow-image
./create_initrd.sh ${PYTHON_SCRIPT}
cd ../../sev_orchestrator
sudo python3 sev-orchestrator.py --initrd ../evaluation/tensorflow-image/${PYTHON_SCRIPT}_initramfs
