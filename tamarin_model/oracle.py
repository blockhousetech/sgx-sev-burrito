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
import re

lines = sys.stdin.readlines()
lemma = sys.argv[1]

maxPriority = 101

rank = []
for i in range(0, maxPriority):
  rank.append([])

if re.match('.*lm_burrito_quote_integrity.*', lemma):
  for line in lines:
    num = line.split(':')[0]
    if re.match('.*TO_Enclave_Generate_Report_For_VM.*', line): rank[100].append(num)
    elif re.match('.*TO_Enclave_VM_Provisioned.*', line): rank[99].append(num)
    elif re.match('.*KU\( ~cik.*', line): rank[98].append(num)
    elif re.match('.*~cik.*', line): rank[97].append(num)
    elif re.match('.*Compromise_SEV_PSP.*', line): rank[96].append(num)
    elif re.match('.*\'transport_keys\'.*', line): rank[95].append(num)
    elif re.match('.*KU\( sign\(\<\'sgx_quote\'.*', line): rank[95].append(num)
    elif re.match('.*!PSP_\'.*', line): rank[94].append(num)
    elif re.match('.*!Intel_Rot\'.*', line): rank[93].append(num)
    elif re.match('.*KU\( ~amd_rot_ltk.*', line): rank[83].append(num)
    elif re.match('.*KU\( ~intel_rot_ltk.*', line): rank[83].append(num)
    elif re.match('.*splitEqs.*', line): rank[12].append(num)
    elif re.match('.*KU\( sign.*\^.*', line): rank[11].append(num)
    elif re.match('.*KU\(.*\^.*', line): rank[10].append(num)
    else: rank[20].append(num)

## UNEXPECTED LEMMAS
else:
  print("Unexpected lemma: {}".format(lemma))
  exit(0);

for goalList in reversed(rank):
  for goal in goalList:
    sys.stderr.write(goal)
    print(goal)
