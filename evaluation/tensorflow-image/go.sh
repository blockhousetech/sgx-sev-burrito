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
source .env
echo "Executing ${PYTHON_SCRIPT}" > stdout.txt 2>&1
python3 ${PYTHON_SCRIPT} >> stdout.txt 2>&1
echo "Script finished with result $?" >> stdout.txt 2>&1
tar -czf model.tar.gz model
echo "Tar finished with result $?" >> stdout.txt 2>&1
python3 sev-report-generation.py --stdout-file stdout.txt --model-file model.tar.gz > report_time.txt 
scp stdout.txt ${SEV_HOST_USER}@${SEV_HOST_IP}:~/eval-tf/${PYTHON_SCRIPT}_stdout.txt
scp model.tar.gz ${SEV_HOST_USER}@${SEV_HOST_IP}:~/eval-tf/${PYTHON_SCRIPT}_model.tar.gz
scp quote.dat ${SEV_HOST_USER}@${SEV_HOST_IP}:~/eval-tf/${PYTHON_SCRIPT}_quote.dat
scp report_time.txt ${SEV_HOST_USER}@${SEV_HOST_IP}:~/eval-tf/${PYTHON_SCRIPT}_report_time.txt
