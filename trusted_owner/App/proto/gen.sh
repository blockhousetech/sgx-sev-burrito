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

/usr/local/bin/protoc --grpc_out=gen/ --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` server.proto
/usr/local/bin/protoc --cpp_out=gen/ server.proto
python3 -m grpc_tools.protoc -I. --python_out=gen/ --grpc_python_out=gen/ server.proto