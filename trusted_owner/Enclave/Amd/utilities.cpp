/*
 * Burrito
 * Copyright (C) 2023 The Blockhouse Technology Limited (TBTL)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "utilities.h"
#include <climits>
#include <cstring>      // memcpy
#include <time.h>
#include "sgx_trts.h"


void sev::gen_random_bytes(void* bytes, size_t num_bytes)
{
    size_t num_gen_bytes = 0;
    num_gen_bytes = sgx_read_rand((unsigned char*)bytes, num_bytes);
}

bool sev::str_to_array(const std::string in_string, uint8_t *array,
                       uint32_t array_size)
{
    std::string substring = "";

    if (array_size < in_string.size() / 2) {
        return false;
    }

    for (size_t i = 0; i < in_string.size()/2; i++) {
        substring = in_string.substr(i*2, 2);
        array[i] = (uint8_t)strtol(substring.c_str(), NULL, 16);
    }

    return true;
}

bool sev::reverse_bytes(uint8_t *bytes, size_t size)
{
    uint8_t *start = bytes;
    uint8_t *end = bytes + size - 1;

    if (!bytes)
        return false;

    while (start < end) {
        uint8_t byte = *start;
        *start = *end;
        *end = byte;
        start++;
        end--;
    }

    return true;
}
