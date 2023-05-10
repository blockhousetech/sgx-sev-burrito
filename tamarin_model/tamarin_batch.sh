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
		--auto-sources \
		$* \
	| sed -E -e 's/(analysis incomplete)/\x1b[94m\1\x1b[0m/g' -e 's/(verified)/\x1b[32m\1\x1b[0m/g' -e 's/(falsified)/\x1b[91m\1\x1b[0m/g'
