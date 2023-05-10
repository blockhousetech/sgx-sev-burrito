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
import base64
import hashlib
import grpc
from argparse import ArgumentParser
import os
from multiprocessing import Process
import threading
from deps.qemu.python.qemu import qmp
import subprocess
from io import StringIO
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))
import calculate_digest
sys.path.append(os.path.join(os.path.dirname(__file__), '../trusted_owner/App/proto/gen'))
from server_pb2 import DeployVmReply, DeployVmRequest, ProvisionVmReply, ProvisionVmRequest
from server_pb2_grpc import TrustedOwnerStub
from decouple import config

SGX_HOST_IP = config("SGX_HOST_IP")
SGX_HOST_GRPC_PORT = config("SGX_HOST_GRPC_PORT")
TO_ADDRESS=SGX_HOST_IP+':'+SGX_HOST_GRPC_PORT
OVMF_FILE = config("OVMF_FILE")
KERNEL_FILE = config("KERNEL_FILE")
CMDLINE = config("CMDLINE", cast=str)
QMP_SOCKET=("localhost",5503)
QEMU_CMD='/usr/local/bin/qemu-system-x86_64 -enable-kvm -cpu EPYC -machine q35 -smp 8,maxcpus=64 -m 32000M,slots=5,maxmem=56G \
      -drive if=pflash,format=raw,unit=0,file={OVMF_FILE},readonly \
      -qmp tcp::5503,server,nowait \
      -S \
      -netdev user,id=vmnic -device e1000,netdev=vmnic,romfile= \
      -machine memory-encryption=sev0,vmport=off \
      -object sev-guest,id=sev0,cbitpos=47,reduced-phys-bits=1,session-file=data/tmp/launch_blob.base64,dh-cert-file=data/tmp/godh.base64,kernel-hashes=on \
      -kernel {KERNEL_FILE} \
      -initrd {INITRD} \
      -append "{CMDLINE}" \
      -vnc :3 -monitor pty -monitor unix:monitor,server,nowait -pidfile qemupid -daemonize'

def read_bytes_from_file(filename):
    with open(filename, 'rb') as file:
        return file.read()


def set_certs(request):
    request.certs.ark = read_bytes_from_file("data/certs/ark.cert")
    request.certs.ask = read_bytes_from_file("data/certs/ask.cert")
    request.certs.cek = read_bytes_from_file("data/certs/cek.cert")
    request.certs.oca = read_bytes_from_file("data/certs/oca.cert")
    request.certs.pek = read_bytes_from_file("data/certs/pek.cert")
    request.certs.pdh = read_bytes_from_file("data/certs/pdh.cert") 

def set_measurement_info(request, initrd):
    request.info.api_major = 0
    request.info.api_minor = 24
    request.info.build_id = 6
    request.info.policy = 1
    request.info.digest = calculate_digest.calculate_digest(initrd)

def call_deploy_vm(client, initrd):
    request = DeployVmRequest()
    set_certs(request)
    set_measurement_info(request, initrd)
    print("Calling Deploy Vm")
    try:
        timer_start = time.perf_counter()
        response = client.DeployVm(request)
        timer_end = time.perf_counter()
        print("Received response: ", response)
        print("Deploy: time elapsed in seconds: ", timer_end-timer_start)
        with open("data/tmp/launch_blob.bin", 'wb') as launch_blob:
            launch_blob.write(response.session_buffer)
        with open("data/tmp/godh.cert", 'wb') as godh_blob:
            godh_blob.write(response.godh_cert)
        with open("data/tmp/launch_blob.base64", 'wb') as launch_blob:
            launch_blob.write(base64.b64encode(response.session_buffer))
        with open("data/tmp/godh.base64", 'wb') as godh_blob:
            godh_blob.write(base64.b64encode(response.godh_cert))
    except grpc.RpcError as e:
        print(e.code())
    else:
       print(grpc.StatusCode.OK)

def setup_vm(initrd):
    proc = subprocess.Popen(QEMU_CMD.format(INITRD = args.initrd, OVMF_FILE = OVMF_FILE, KERNEL_FILE = KERNEL_FILE, CMDLINE = CMDLINE), shell=True)
    proc.wait()


def qemu_query_launch_measure():
    print("Querying launch measure")
    Qmp = qmp.QEMUMonitorProtocol(address=QMP_SOCKET)
    Qmp.connect()
    launch_measure_b64 = Qmp.command('query-sev-launch-measure')
    Qmp.close()
    launch_measure_buffer = base64.b64decode(launch_measure_b64["data"])
    launch_measure = launch_measure_buffer[0:32]
    mnonce = launch_measure_buffer[32:48]
    return (launch_measure, mnonce)

def call_provision_vm(client):
    request = ProvisionVmRequest()
    launch_measure, mnonce = qemu_query_launch_measure()
    request.measurement = launch_measure
    print("Launch measure:",request.measurement.hex())
    request.mnonce = mnonce
    print("Mnonce:",request.mnonce.hex())    
    print("Calling Provision Vm")
    timer_start = time.perf_counter()
    response = client.ProvisionVm(request)
    timer_end = time.perf_counter()
    print("Received response:", response)
    print("Provision: time elapsed in seconds: ", timer_end-timer_start)
    with open("data/tmp/secret_header.dat", 'wb') as secret_header:
        secret_header.write(response.secret_header)
    with open("data/tmp/secret_blob.dat", 'wb') as secret_blob:
        secret_blob.write(response.secret_blob)
    return response
    

def provision_and_start_vm(response):
    print("Provisioning and starting Vm")
    print("HEADER =",base64.b64encode(response.secret_header).decode())
    print("HEADER HEX =",response.secret_header.hex())
    print("SECRET =",base64.b64encode(response.secret_blob).decode())
    print("SECRET HEX =",response.secret_blob.hex())
    print("SERA =",25)

    Qmp = qmp.QEMUMonitorProtocol(address=QMP_SOCKET)
    Qmp.connect()
    Qmp.command('sev-inject-launch-secret',
                **{'packet-header': base64.b64encode(response.secret_header).decode(),
                'secret': base64.b64encode(response.secret_blob).decode()})

    print('\nProvisioning successful, starting VM')
    Qmp.command('cont')
    Qmp.close()

def call_provision_and_provision_and_start_vm():
    with grpc.insecure_channel(TO_ADDRESS) as channel:
        print("Connected 2")
        client = TrustedOwnerStub(channel)
        secret_response = call_provision_vm(client)
        provision_and_start_vm(secret_response)

def find_qemu_detached_process_pid():
    detached_pid = "0"
    with open("qemupid",'r') as pidfile:
        detached_pid = pidfile.readline().strip()
    assert detached_pid != "0"
    print("Detached qemu pid:", detached_pid)
    return detached_pid

def wait_for_detached_process_with_pid(detached_pid):
    os.system("tail --pid={} -f /dev/null".format(detached_pid))

if __name__ == "__main__":
    parser = ArgumentParser(description='Orchestrate SEV deployment with SGX protocol')
    parser.add_argument('--initrd',
                        help='Initramfs file',
                        required=True)
    args = parser.parse_args()

    with grpc.insecure_channel(TO_ADDRESS) as channel:
        print("Connected")
        client = TrustedOwnerStub(channel)
        call_deploy_vm(client, args.initrd)
    print("Executing: ", QEMU_CMD.format(INITRD = args.initrd, OVMF_FILE = OVMF_FILE, KERNEL_FILE = KERNEL_FILE, CMDLINE = CMDLINE))
    timer_start = time.perf_counter()
    setup_vm(args.initrd)
    detached_pid = find_qemu_detached_process_pid()
    #setup_vm(args.initrd)
    #time.sleep(0.01490) # This is how long it takes so that qmp is ready to connect
    proc = Process(target=call_provision_and_provision_and_start_vm)
    proc.start()
    wait_for_detached_process_with_pid(detached_pid)
    #secret_response = call_provision_vm(client)
    #provision_and_start_vm(secret_response)
    #p.join()
    timer_end = time.perf_counter()
    print("\nVM lifespan in seconds: ", timer_end-timer_start)
