#!/usr/bin/python3
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
import sys
import os 
import base64
import hmac
import hashlib
from argparse import ArgumentParser
from uuid import UUID
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binascii
from decouple import config

OVMF_FILE = config("OVMF_FILE")
KERNEL_FILE = config("KERNEL_FILE")
CMDLINE = config("CMDLINE", cast=str)+'\00'

def calculate_digest(initrd_file):
    with open (OVMF_FILE, 'rb') as fh:
        ovmf_file = fh.read()
        ovmf_hash_object = hashlib.sha256(ovmf_file)
        ovmf_hash = ovmf_hash_object.digest()
        ovmf_hash_length = ovmf_hash_object.digest_size
        # Hardcoded list of sev hashtable UUIDs
        hashtable_config_guid = UUID('{7255371f-3a3b-4b04-927b-1da6efa8d454}').bytes_le
        hashtable_guid = UUID('{9438d606-4f22-4cc9-b479-a793d411fd21}').bytes_le
        kernel_guid = UUID('{4de79437-abd2-427f-b835-d5b172d2045b}').bytes_le
        initrd_guid = UUID('{44baf731-3a2f-4bd7-9af1-41e29169781d}').bytes_le
        cmdline_guid = UUID('{97d02dd8-bd20-4c94-aa78-e7714d36ab2a}').bytes_le

        if hashtable_config_guid in ovmf_file:
            with open (KERNEL_FILE, 'rb') as fh:
                kernel_file = fh.read()
                kernel_hash_object = hashlib.sha256(kernel_file)
                kernel_hash = kernel_hash_object.digest()
                kernel_hash_length = kernel_hash_object.digest_size

            with open (initrd_file, 'rb') as fh:
                initrd_file = fh.read()
                initrd_hash_object = hashlib.sha256(initrd_file)
                initrd_hash = initrd_hash_object.digest()
                initrd_hash_length = initrd_hash_object.digest_size

            cmdline_hash_object = hashlib.sha256(CMDLINE.encode('utf-8'))
            cmdline_hash = cmdline_hash_object.digest()
            cmdline_hash_length = cmdline_hash_object.digest_size

            # Calculate sizes for designated areas in hasbtable
            # size 2 is designated for the size of each area (it is the size of subarea that holds the size of each area)
            hashtable_kernel_area_length = len(kernel_guid) + 2 + kernel_hash_length
            hashtable_initrd_area_length = len(initrd_guid) + 2 + initrd_hash_length
            hashtable_cmdline_area_length = len(cmdline_guid) + 2 + cmdline_hash_length
            hashtable_area_length = len(hashtable_config_guid) + 2 + hashtable_kernel_area_length + hashtable_initrd_area_length + hashtable_cmdline_area_length

            #padded hashtable size is round up to a multiple of 16
            padding_length = 16 - (hashtable_area_length % 16)

            
            # creata a new ovmf, appending the hashes
            new_ovmf = bytearray(ovmf_file)
            new_ovmf.extend(hashtable_guid + (hashtable_area_length).to_bytes(2, byteorder='little') + cmdline_guid + (hashtable_cmdline_area_length).to_bytes(2, byteorder='little') + cmdline_hash + initrd_guid + (hashtable_initrd_area_length).to_bytes(2, byteorder='little') + initrd_hash + kernel_guid + (hashtable_kernel_area_length).to_bytes(2, byteorder='little') + kernel_hash + bytearray(padding_length))

            # substitute the old ovmf file with the new one
            ovmf_file = bytes(new_ovmf)
            ovmf_hash_object = hashlib.sha256(ovmf_file)
            ovmf_hash = ovmf_hash_object.digest()
            ovmf_hash_length = ovmf_hash_object.digest_size

            print("VM Digest is: ", ovmf_hash.hex())
            return ovmf_hash
        else:
            parser.error('hashtable designated area not found in ovmf file')

if __name__ == "__main__":
    parser = ArgumentParser(description='Calculate SEV VM digest')
    parser.add_argument('--initrd',
                        help='location of initrd file to calculate the hash from',
                        required=True)
    args = parser.parse_args()
    calculate_digest(args.initrd)

        
