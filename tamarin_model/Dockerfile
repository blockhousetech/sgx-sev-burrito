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
FROM archlinux:latest
RUN echo "Server=https://archive.archlinux.org/repos/2021/11/10/\$repo/os/\$arch" > /etc/pacman.d/mirrorlist
RUN pacman -Sy --noconfirm
RUN pacman -S --noconfirm tamarin-prover
RUN pacman -S --noconfirm python3
ENV PATH /usr/share/maude:$PATH
RUN cp /usr/bin/maude /usr/share/maude/
CMD ["tamarin-prover", "interactive", ".", "--interface=*4", "--auto-sources", "--quit-on-warning"]
