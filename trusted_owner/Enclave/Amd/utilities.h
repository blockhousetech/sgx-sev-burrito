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

#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>

namespace sev
{
    #define SEV_DEFAULT_DIR       "/usr/psp-sev-assets/"
    #define KDS_CERT_SITE         "https://kdsintf.amd.com"
    #define KDS_DEV_CERT_SITE     "https://kdsintfdev.amd.com"
    #define KDS_CEK               KDS_CERT_SITE "/cek/id/"
    #define KDS_VCEK              KDS_CERT_SITE "/vcek/v1/"   // KDS_VCEK/{product_name}/{hwid}?{tcb parameter list}
    #define KDS_VCEK_CERT_CHAIN   "cert_chain"                // KDS_VCEK/{product_name}/cert_chain
    #define KDS_VCEK_CRL          "crl"                       // KDS_VCEK/{product_name}/crl"

    #define PAGE_SIZE               4096        // Todo remove this one?
    #define PAGE_SIZE_4K            4096
    #define PAGE_SIZE_2M            (512*PAGE_SIZE_4K)

    #define IS_ALIGNED(e, x)            (0==(((uintptr_t)(e))%(x)))
    #define IS_ALIGNED_TO_16_BYTES(e)   IS_ALIGNED((e), 16)         // 4 bits
    #define IS_ALIGNED_TO_32_BYTES(e)   IS_ALIGNED((e), 32)         // 5 bits
    #define IS_ALIGNED_TO_64_BYTES(e)   IS_ALIGNED((e), 64)         // 6 bits
    #define IS_ALIGNED_TO_128_BYTES(e)  IS_ALIGNED((e), 128)        // 7 bits
    #define IS_ALIGNED_TO_4KB(e)        IS_ALIGNED((e), 4096)       // 12 bits
    #define IS_ALIGNED_TO_1MB(e)        IS_ALIGNED((e), 0x100000)   // 20 bits
    #define IS_ALIGNED_TO_2MB(e)        IS_ALIGNED((e), 0x200000)   // 21 bits

    #define ALIGN_TO_16_BYTES(e)        ((((uintptr_t)(e))+0xF)&(~(uintptr_t)0xF))
    #define ALIGN_TO_32_BYTES(e)        ((((uintptr_t)(e))+0x1F)&(~(uintptr_t)0x1F))
    #define ALIGN_TO_64_BYTES(e)        ((((uintptr_t)(e))+0x3F)&(~(uintptr_t)0x3F))

    #define BITS_PER_BYTE    8

    /**
     * Generate some random bytes
     */
    void gen_random_bytes(void *bytes, size_t num_bytes);

    /**
     * Converts a string of ascii-encoded hex bytes into a Hex array
     * Ex. To generate the string, do printf("%02x", myArray) will generate
     *     "0123456ACF" and this function will put it back into an array
     * This function is expecting the input string to be an even number of
     *      elements not including the null terminator
     */
    bool str_to_array(const std::string in_string, uint8_t *array, uint32_t array_size);

    /**
     * Reverses bytes in a section of memory. Used in validating cert signatures
     */
    bool reverse_bytes(uint8_t *bytes, size_t size);

} // namespace

#endif /* UTILITIES_H */
