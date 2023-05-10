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

from asyncore import file_dispatcher
import sys, getopt
import hmac
import hashlib
import grpc
import time
from argparse import ArgumentParser
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../trusted_owner/App/proto/gen'))
from server_pb2 import GenerateReportForVmReply, GenerateReportForVmRequest
from server_pb2_grpc import TrustedOwnerStub


TO_ADDRESS='20.234.100.189:50051'
SECRET_ADRESS='/secrets/coco/736869e5-84f0-4973-92ec-06879ce3da0b'

def get_key():
    with open(SECRET_ADRESS, 'rb') as file:
        key = file.read()
        return key

def generate_hmac(key, data):
    print("key="+key.hex())
    h = hmac.new(key, data, hashlib.sha256)
    print(h.hexdigest())
    return h.digest()

def generate_vm_data_hmac(data):
    key = get_key()
    return generate_hmac(key, data)


def generate_report_trusted_vm_owner(data):
    meas = generate_vm_data_hmac(data)
    print("hmac="+meas.hex()+'\n')
    with grpc.insecure_channel(TO_ADDRESS) as channel:
        client = TrustedOwnerStub(channel)
        print("data="+data.hex())
        print("hmac="+meas.hex())
        request = GenerateReportForVmRequest(vm_data=data,vm_data_hmac=meas)
        timer_start = time.perf_counter()
        response = client.GenerateReportForVm(request)
        timer_end = time.perf_counter()
        print("Report: time elapsed in seconds: ", timer_end-timer_start, flush=True)
        with open("quote.dat", 'wb') as quote_file:
            quote_file.write(response.quote)
        



if __name__ == "__main__":
    parser = ArgumentParser(description='Inject secret into SEV')
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
    print("Data="+tensorflow_files_hash.hex()+'\n')

    generate_report_trusted_vm_owner(tensorflow_files_hash)
    
