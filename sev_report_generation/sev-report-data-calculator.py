#!/usr/bin/python
#
# Burrito
# Copyright (C) 2023 The Blockhouse Technology Limited (TBTL)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import sys, getopt
import hashlib
from argparse import ArgumentParser
import base64


if __name__ == "__main__":
    parser = ArgumentParser(description='Data calculation')
    parser.add_argument('--stdout-file',
                        help='Tensorflow script stdout file',
                        required=True)
    parser.add_argument('--model-file',
                        help='Tensorflow script exported model file',
                        required=True)
    args = parser.parse_args()

    
    hasher = hashlib.sha256()
    with open(args.stdout_file, 'rb') as stdout_file:
        hasher.update(stdout_file.read())
    with open(args.model_file, 'rb') as model_file:
        hasher.update(model_file.read())
    tensorflow_files_hash = hasher.digest()
    tensorflow_files_hash += b'\0'*32 
    print("Data in hex is",tensorflow_files_hash.hex())
    print("Data in base64 is",base64.b64encode(tensorflow_files_hash).decode())
    