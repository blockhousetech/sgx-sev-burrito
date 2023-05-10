<!--
Burrito
Copyright (C) 2023 The Blockhouse Technology Limited (TBTL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
-->
Tamarin model of the system
===========================

To build the docker image containing the Tamarin prover: 

`./tamarin_build.sh`

Edit the scripts to reflect the amount of RAM availble on the verification machine.

To reproduce the results run the following command. On a typical desktop macine, it should take under a minute to reproduce the results and under an hour to reproduce all the executable and sanity checking lemmas.

`./reproduce_results.sh`

`./reproduce_sanity_checks.sh`

To run the verification in interactive mode run the command below and navigate to http://127.0.0.1:3001.

`./tamarin_batch.sh ./burrito.spthy --prove`
