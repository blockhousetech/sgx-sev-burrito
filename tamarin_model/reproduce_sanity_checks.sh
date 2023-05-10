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

set -e

HASKELL_MEMORY=22G
DOCKER_MEMORY=24G

time docker run -it -v ${PWD}:/src -w /src --memory=${DOCKER_MEMORY} tamarin-arch \
	tamarin-prover \
		+RTS -M${HASKELL_MEMORY} -RTS \
		--quit-on-warning \
		--prove=should_fail* \
		--prove=ex* \
		--output=burrito_sanity_checks.spthy \
		--defines=SIMPLE \
		--stop-on-trace=SEQDFS \
		burrito.spthy \
	> burrito_sanity_checks_output.txt

cat burrito_sanity_checks_output.txt | grep "summary of summaries:" -B 1 -A 1000 > burrito_sanity_checks_summary.txt
