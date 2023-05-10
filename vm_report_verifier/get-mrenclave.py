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
import os
from argparse import ArgumentParser
import base64

if __name__ == "__main__":
    parser = ArgumentParser(description='Get mrenclave')
    parser.add_argument('--signed-enclave-file',
                        help='Signed enclave file',
                        required=True)
    args = parser.parse_args()

    print("Finding mrenclave for", args.signed_enclave_file)
    cmd = "sgx_sign dump -enclave {ENCLAVE_FILE} -dumpfile dump.txt".format(ENCLAVE_FILE = args.signed_enclave_file)
    print("Executing:", cmd)
    os.system(cmd)

    with open("dump.txt", 'r') as file:
        line = file.readline()
        while line:
            if line == "metadata->enclave_css.body.enclave_hash.m:":
                mr = file.readline().replace(" ","").replace("0x","").strip()+file.readline().replace(" ","").replace("0x","").strip()
                print("Mrenclave is", mr)
                print("Mrenclave in hex is", bytes.fromhex(mr).hex())
                print("Mrenclave in base64 is", base64.b64encode(bytes.fromhex(mr)).decode())
                break
            else:
                line = file.readline().strip()